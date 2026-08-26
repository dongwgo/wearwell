"""치수 -> 체형 레퍼런스 이미지.

기존 아바타 생성의 근본 문제는 치수를 텍스트로만 넘긴다는 점이다.
"shoulder width 46 cm, waist circumference 80 cm"를 프롬프트에 써도 확산 모델은
숫자를 길이로 해석하지 못한다. 실제로 프롬프트의 숫자만 바꿔가며 생성해 보면
결과 체형은 거의 같고, 바뀌는 건 얼굴과 조명 정도다. 즉 **측정 불가능한**
파이프라인이다.

이 모듈은 숫자를 이미지로 바꾼다.

    치수 -> SMPL-X beta 최적화 -> 메시 -> 정면 실루엣 렌더 -> FLUX 참조 이미지 1

이렇게 하면 두 가지가 생긴다.
1. 확산 모델이 따라갈 **구조적 조건**. 텍스트보다 훨씬 강하게 먹는다.
2. **검증 가능한 수치**. 맞춰진 메시를 다시 측정해 목표 치수와의 오차를 cm로
   찍을 수 있다. 발표의 "수치 기반 개선"이 여기서 나온다.

SMPL-X 가중치는 https://smpl-x.is.tue.mpg.de 에서 등록 후 받아야 한다(무료,
연구용). 가중치가 없으면 인체 계측 비율로 만든 2D 실루엣으로 자동 폴백해서
서비스는 계속 돌아간다 — 이때도 키 대비 비율은 목표 치수를 따른다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SMPLX_MODEL_PATH = os.getenv("SMPLX_MODEL_PATH", "")
RENDER_SIZE = (768, 1152)  # app.py의 INFERENCE_SIZE와 같아야 한다

# 아바타를 돌려 볼 수 있는 시점. 정면 기준 Y축 회전각(도)이다.
# SMPL-X 경로는 이 각도로 실제 메시를 돌려 렌더하고, 폴백 경로는 같은 치수에서
# 단면 깊이를 써서 그 시점의 실루엣을 그린다.
VIEW_YAW = {"front": 0.0, "side": 90.0, "back": 180.0}
DEFAULT_VIEW = "front"

# 프레임 안에서 몸이 차지하는 세로 비율. 위아래로 여백을 남겨야 확산 모델이
# 머리 위와 발밑을 잘라먹지 않는다. 0.9를 넘기면 발이 잘리기 시작한다.
BODY_FRAME_RATIO = 0.86

# 최적화가 맞출 치수. 사용자가 입력하지 않은 항목은 목표에서 빠진다.
FITTED_MEASUREMENTS = ("height", "chest", "waist", "hip", "shoulder")


@dataclass
class BodyTarget:
    gender: str  # "men" | "women"
    height: float  # cm
    weight: float  # kg
    chest: float | None = None
    waist: float | None = None
    hip: float | None = None
    shoulder: float | None = None
    inseam: float | None = None

    def goals(self) -> dict[str, float]:
        """실제로 최적화에 넣을 목표만 추린다."""
        estimated = estimate_missing(self)
        return {key: value for key, value in estimated.items() if value is not None}


@dataclass
class BodyReference:
    image: Image.Image
    achieved: dict[str, float] = field(default_factory=dict)  # 맞춘 몸의 실측치 (cm)
    target: dict[str, float] = field(default_factory=dict)
    betas: list[float] | None = None
    source: str = "silhouette-fallback"
    view: str = DEFAULT_VIEW

    def errors(self) -> dict[str, float]:
        """목표 대비 오차 (cm). 개선 전후 비교에 그대로 쓴다."""
        return {
            key: round(self.achieved[key] - value, 1)
            for key, value in self.target.items()
            if key in self.achieved
        }

    def mean_absolute_error(self) -> float:
        errors = self.errors()
        return round(sum(abs(v) for v in errors.values()) / len(errors), 2) if errors else 0.0


def estimate_missing(target: BodyTarget) -> dict[str, float | None]:
    """입력하지 않은 둘레는 BMI 기반 회귀식으로 메운다.

    사용자가 가슴·허리·엉덩이를 다 재서 넣는 경우는 거의 없다. 비워두면
    최적화가 그 축을 자유롭게 놔둬서 매번 다른 체형이 나오므로, 없는 값은
    키·몸무게에서 추정해 채워 넣고 최적화를 안정시킨다.
    추정치는 근사이며 실제 신체 계측이나 의류 사이즈 판정이 아니다.
    """
    bmi = target.weight / ((target.height / 100) ** 2)
    women = target.gender == "women"
    return {
        "height": target.height,
        # 계수는 성인 한국인 표준 인체치수(Size Korea) 평균 비율을 단순 선형화한 값.
        "chest": target.chest or round(target.height * (0.49 if women else 0.52) + (bmi - 21.5) * 1.65, 1),
        "waist": target.waist or round(target.height * (0.41 if women else 0.45) + (bmi - 21.5) * 2.05, 1),
        "hip": target.hip or round(target.height * (0.55 if women else 0.53) + (bmi - 21.5) * 1.35, 1),
        "shoulder": target.shoulder or round(target.height * (0.225 if women else 0.245), 1),
        "inseam": target.inseam or round(target.height * (0.455 if women else 0.46), 1),
    }


# --- SMPL-X 경로 -----------------------------------------------------------


def _slice_circumference(vertices: np.ndarray, y: float, tolerance: float = 0.012) -> float:
    """y 높이의 수평 단면 둘레 (m).

    해당 높이 근처 정점을 모아 xz 평면에 투영하고 convex hull 둘레를 잰다.
    줄자를 몸에 감았을 때와 같은 값이 나온다 — 줄자는 오목한 곳을 파고들지
    않고 볼록한 지점 사이를 가로지르기 때문에 convex hull이 맞다.
    """
    band = vertices[np.abs(vertices[:, 1] - y) < tolerance]
    if len(band) < 8:
        return 0.0
    points = band[:, [0, 2]]
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        ordered = points[hull.vertices]
    except Exception:
        pass  # scipy가 없으면 각도 정렬 폴리곤으로 근사
    closed = np.vstack([ordered, ordered[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def measure_mesh(vertices: np.ndarray) -> dict[str, float]:
    """SMPL-X 정점 (N,3) -> 인체 치수 (cm).

    둘레를 재는 높이는 메시 자체의 높이 비율로 잡는다. 고정된 y좌표를 쓰면
    키가 다른 체형에서 허리 대신 갈비뼈를 재게 된다.
    """
    y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    height_m = y_max - y_min

    def at(ratio: float) -> float:
        return y_min + height_m * ratio

    # 비율은 성인 인체 표준 landmark 높이(키 대비).
    chest = _slice_circumference(vertices, at(0.720))
    waist = _slice_circumference(vertices, at(0.620))
    hip = _slice_circumference(vertices, at(0.515))

    # 어깨 너비는 둘레가 아니라 좌우 최대폭 — 가슴 높이보다 살짝 위에서 잰다.
    band = vertices[np.abs(vertices[:, 1] - at(0.805)) < 0.02]
    shoulder = float(band[:, 0].max() - band[:, 0].min()) if len(band) else 0.0

    return {
        "height": round(height_m * 100, 1),
        "chest": round(chest * 100, 1),
        "waist": round(waist * 100, 1),
        "hip": round(hip * 100, 1),
        "shoulder": round(shoulder * 100, 1),
    }


def fit_betas(target: BodyTarget, num_betas: int = 10, max_iterations: int = 60):
    """목표 치수에 맞는 SMPL-X shape 파라미터를 찾는다.

    beta는 PCA 축이라 치수와 1:1로 대응하지 않는다. 그래서 해석적으로 풀지
    못하고, 메시를 만들어 재고 -> 오차를 줄이는 최소자승 반복으로 푼다.
    beta 10개 / 목표 5개짜리 작은 문제라 CPU로 몇 초면 끝난다.
    """
    import torch
    from scipy.optimize import least_squares

    import smplx

    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="female" if target.gender == "women" else "male",
        num_betas=num_betas,
        use_pca=False,
    )
    goals = target.goals()
    keys = [key for key in FITTED_MEASUREMENTS if key in goals]

    def vertices_for(betas: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            output = model(betas=torch.tensor(betas[None], dtype=torch.float32))
        return output.vertices[0].numpy()

    def residual(betas: np.ndarray) -> np.ndarray:
        measured = measure_mesh(vertices_for(betas))
        # cm 단위 오차를 그대로 쓰면 키(170)가 어깨(45)를 압도한다. 목표값으로
        # 나눠 상대 오차로 맞춰야 모든 축이 고르게 수렴한다.
        return np.array([(measured[key] - goals[key]) / goals[key] for key in keys])

    start = np.zeros(num_betas)
    solution = least_squares(
        residual,
        start,
        bounds=(-4.0, 4.0),  # SMPL-X beta는 ±4σ를 넘으면 인체가 아니게 된다
        max_nfev=max_iterations,
        xtol=1e-4,
    )
    vertices = vertices_for(solution.x)
    return solution.x, vertices, measure_mesh(vertices)


def _edge_colour(pixels: np.ndarray, rows: slice, keep_fraction: float) -> tuple[int, int, int]:
    """가장자리 띠에서 배경색을 고른다.

    몸이 서 있는 가운데 열을 빼고 바깥쪽만 본다. 아래쪽 띠는 가운데에 다리와
    신발이 있어서, 통째로 평균 내면 배경이 아니라 살색이 나온다.
    """
    band = pixels[rows]
    width = band.shape[1]
    margin = max(1, int(width * keep_fraction))
    outer = np.concatenate([band[:, :margin], band[:, width - margin:]], axis=1)
    return tuple(int(v) for v in np.median(outer.reshape(-1, 3), axis=0))


def has_room_below(image: Image.Image, threshold: float = 0.012) -> bool:
    """아래쪽에 이미 빈 배경이 있는가.

    맨 아랫줄이 배경 한 가지 색으로 균일하면 인물이 프레임 안에서 끝났다는
    뜻이고, 여백을 더 붙일 이유가 없다. 실제 사진처럼 바닥·가구가 이어지는
    경우에는 색이 흩어지므로 여백을 붙여 준다.
    """
<<<<<<< Updated upstream
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    strip = pixels[-max(2, int(pixels.shape[0] * 0.02)):]
=======
    width, height = image.size
    rows = max(2, int(height * 0.02))
    # 먼저 자르고 변환한다. 전체를 float32로 복사하면 4000px 사진 한 장에
    # 180MB가 순간적으로 잡힌다 — 아래 2%만 보면 되는 일이다.
    strip = np.asarray(image.convert("RGB").crop((0, height - rows, width, height)), dtype=np.float32)
>>>>>>> Stashed changes
    return float(strip.reshape(-1, 3).std(axis=0).mean()) / 255.0 < threshold


def fit_generation_size(image: Image.Image, budget=RENDER_SIZE, step: int = 16) -> tuple[int, int]:
    """원본 비율을 지키면서 비슷한 픽셀 수를 갖는 생성 크기.

    출력 크기를 768x1152로 못 박아 두면 3:4 사진이 2:3으로 늘어난다 —
    실제 사진에 옷을 입히면 사람과 배경이 세로로 길쭉해지는 이유가 이것이다.
    확산 모델은 보통 16의 배수 크기를 요구하므로 그 격자에 맞춰 반올림한다.
    """
    width, height = image.size
    if not width or not height:
        return budget
    pixels = budget[0] * budget[1]
    scale = math.sqrt(pixels / (width * height))
    fitted = (
        max(step, int(round(width * scale / step)) * step),
        max(step, int(round(height * scale / step)) * step),
    )
    return fitted


<<<<<<< Updated upstream
=======
def downscale_to_budget(image: Image.Image, budget=RENDER_SIZE, slack: float = 1.6) -> Image.Image:
    """생성 크기보다 한참 큰 사진은 미리 줄인다.

    휴대폰 사진은 4000px가 예사고, 그대로 들고 다니면 패딩·복사 단계마다
    수백 MB가 잡힌다. 어차피 출력은 생성 크기라 원본 해상도는 쓰이지 않는다.
    """
    limit = budget[0] * budget[1] * slack
    pixels = image.width * image.height
    if pixels <= limit:
        return image
    scale = math.sqrt(limit / pixels)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


>>>>>>> Stashed changes
def pad_for_full_body(
    image: Image.Image, headroom: float = 0.05, footroom: float = 0.12
) -> Image.Image:
    """인물 사진 위아래에 배경색 여백을 덧댄다. 가로세로 비율은 그대로 둔다.

    착장 결과에서 발이 잘리는 문제는 프롬프트만으로 잘 잡히지 않는다 —
    FLUX.2는 참조 이미지의 **구도**를 그대로 따라가는 성향이 강해서, 넣어준
    사람 사진이 종아리에서 끝나면 결과도 종아리에서 끝난다. 문장으로 부탁하는
    대신 참조 이미지에 실제로 빈 공간을 만들어 "여기까지가 화면"이라고 보여준다.

    세로로만 늘린 뒤 원래 크기로 되돌리면 사람이 눌려서 넓어 보인다(실측 17%).
    가로에도 같은 비율로 여백을 붙여 비율을 유지한다.
    """
<<<<<<< Updated upstream
    if has_room_below(image):
        return image.convert("RGB")

    pixels = np.asarray(image.convert("RGB"))
=======
    image = downscale_to_budget(image.convert("RGB"))
    if has_room_below(image):
        return image

    pixels = np.asarray(image)
>>>>>>> Stashed changes
    height, width = pixels.shape[:2]
    strip = max(2, int(height * 0.03))
    top_colour = _edge_colour(pixels, slice(0, strip), 0.30)
    bottom_colour = _edge_colour(pixels, slice(height - strip, height), 0.15)

    top_pad = int(height * headroom)
    bottom_pad = int(height * footroom)
    # 세로가 늘어난 만큼 가로도 늘려야 비율이 유지된다.
    side_pad = int(round(width * (top_pad + bottom_pad) / (2 * height)))

    canvas = Image.new("RGB", (width + 2 * side_pad, height + top_pad + bottom_pad), top_colour)
    draw = ImageDraw.Draw(canvas)
    if bottom_pad:
        draw.rectangle((0, top_pad + height, canvas.width, canvas.height), fill=bottom_colour)
    if side_pad:
        # 좌우 띠는 위아래 색을 세로로 이어 붙여 배경 흐름을 끊지 않는다.
        for x0, x1 in ((0, side_pad), (canvas.width - side_pad, canvas.width)):
            draw.rectangle((x0, 0, x1, top_pad + height // 2), fill=top_colour)
            draw.rectangle((x0, top_pad + height // 2, x1, canvas.height), fill=bottom_colour)
    canvas.paste(image.convert("RGB"), (side_pad, top_pad))
    # 이어붙인 자리에 생기는 경계선을 없앤다. 띠 부분만 흐리게 문질러서 배경이
    # 자연스럽게 이어지도록 하고, 인물 영역은 건드리지 않는다.
    for y0, y1 in ((0, top_pad), (top_pad + height, canvas.height)):
        if y1 - y0 < 4:
            continue
        seam = (0, max(0, y0 - 6), canvas.width, min(canvas.height, y1 + 6))
        canvas.paste(canvas.crop(seam).filter(ImageFilter.GaussianBlur(5)), seam[:2])
    return canvas


def rotate_y(vertices: np.ndarray, degrees: float) -> np.ndarray:
    """메시를 세로축 기준으로 돌린다. 측면·후면 뷰는 이걸로 공짜로 얻는다."""
    if not degrees:
        return vertices
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    return np.stack([cos * x + sin * z, y, -sin * x + cos * z], axis=1)


def render_mesh_silhouette(vertices: np.ndarray, size=RENDER_SIZE, yaw: float = 0.0) -> Image.Image:
    """메시를 정면 직교 투영으로 음영 렌더한다.

    pyrender/OpenGL은 Colab 헤드리스에서 자주 깨지므로 numpy 스플랫으로 그린다.
    FLUX에 넘길 구조 레퍼런스는 실루엣과 깊이 음영이면 충분하고, 삼각형
    래스터라이저까지 갈 필요가 없다.
    """
    width, height = size
    vertices = rotate_y(vertices, yaw)
    y = vertices[:, 1]
    x = vertices[:, 0]
    z = vertices[:, 2]

    # 위아래 여백. 좁게 잡으면 FLUX가 그 프레이밍을 따라 하면서 발끝을 잘라낸다.
    margin = (1 - BODY_FRAME_RATIO) / 2
    scale = (height * (1 - 2 * margin)) / (y.max() - y.min())
    px = np.round(width / 2 + x * scale).astype(int)
    py = np.round(height * (1 - margin) - (y - y.min()) * scale).astype(int)

    depth = np.full((height, width), -np.inf, dtype=np.float32)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    # 같은 픽셀에 여러 정점이 오면 카메라에 가까운(z가 큰) 쪽만 남긴다.
    for xi, yi, zi in zip(px[valid], py[valid], z[valid]):
        if zi > depth[yi, xi]:
            depth[yi, xi] = zi

    filled = np.isfinite(depth)
    shaded = np.zeros((height, width), dtype=np.uint8)
    if filled.any():
        near, far = depth[filled].min(), depth[filled].max()
        span = max(far - near, 1e-6)
        shaded[filled] = (90 + 150 * (depth[filled] - near) / span).astype(np.uint8)

    image = Image.fromarray(shaded, mode="L")
    # 정점 스플랫이라 표면에 구멍이 남는다. 닫고 살짝 흐려 연속면으로 만든다.
    image = image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    image = image.filter(ImageFilter.GaussianBlur(1.2))
    return Image.merge("RGB", (image, image, image))


# --- 폴백 경로 (SMPL-X 가중치 없이) ----------------------------------------


def render_proportional_silhouette(
    target: BodyTarget, size=RENDER_SIZE, view: str = DEFAULT_VIEW
) -> tuple[Image.Image, dict[str, float]]:
    """인체 계측 비율로 만든 2D 실루엣.

    메시가 아니라서 둘레를 직접 잴 수는 없지만, 둘레를 타원 둘레로 역산해
    화면상의 폭을 정하므로 실루엣의 비율은 목표 치수를 따른다.

    측면 뷰는 같은 타원의 **깊이**(앞뒤 두께)를 폭으로 쓴다 — 3D 메시가 없어도
    "옆에서 보면 얼마나 두꺼운가"는 둘레와 앞뒤 비율에서 나온다.
    후면 뷰는 정면과 실루엣이 같으므로 좌우를 뒤집고 얼굴 쪽 힌트만 뺀다.
    """
    width, height = size
    goals = {key: value for key, value in estimate_missing(target).items() if value}
    px_per_cm = (height * BODY_FRAME_RATIO) / target.height
    side_view = view == "side"

    def ellipse_width(circumference: float, depth_ratio: float) -> float:
        """둘레 -> 정면에서 보이는 폭.

        몸통 단면을 타원으로 보고 Ramanujan 근사를 역으로 푼다. 단면을 원으로
        보면(둘레/pi) 몸이 앞뒤로 납작하다는 사실을 무시해 폭이 과소평가된다.
        """
        # C ≈ pi * (3(a+b) - sqrt((3a+b)(a+3b))), b = depth_ratio * a
        r = depth_ratio
        factor = math.pi * (3 * (1 + r) - math.sqrt((3 + r) * (1 + 3 * r)))
        front = 2 * (circumference / factor) * px_per_cm
        # 옆에서 보이는 폭은 타원의 짧은 축, 즉 앞뒤 두께다.
        return front * r if side_view else front

    canvas = Image.new("RGB", (width, height), "#f2efec")
    draw = ImageDraw.Draw(canvas)
    cx = width / 2
    ground = height * (1 - (1 - BODY_FRAME_RATIO) / 2)
    stature = target.height * px_per_cm
    top = ground - stature

    def at(ratio: float) -> float:
        """정수리에서 발끝까지를 0..1로 본 높이. 성인 표준 landmark 비율."""
        return top + stature * ratio

    body = "#8d949b"
    # 실루엣은 키와 무관하게 화면을 꽉 채우므로(전신 카탈로그 컷과 같은 프레이밍),
    # 절대 신장은 픽셀 높이가 아니라 **비율**로만 전달된다. 두신지수는 키에 따라
    # 커지므로(성인 대략 155cm에서 7.0, 190cm에서 8.1) 이 값이 키 신호를 나른다.
    head_index = min(8.1, max(6.9, 7.0 + (target.height - 155) / 35 * 1.1))
    head_h = stature / head_index
    chest_w = ellipse_width(goals["chest"], 0.72)
    waist_w = ellipse_width(goals["waist"], 0.74)
    hip_w = ellipse_width(goals["hip"], 0.78)
    # 옆에서 보면 어깨너비는 보이지 않는다. 대신 어깨 높이의 몸통 두께가 보이고,
    # 그건 가슴 깊이와 거의 같다.
    shoulder_w = chest_w * 1.05 if side_view else goals["shoulder"] * px_per_cm

    y_shoulder, y_chest, y_waist, y_hip, y_crotch = (
        at(0.185), at(0.270), at(0.375), at(0.470), at(0.520),
    )

    # 머리 + 목
    draw.ellipse((cx - head_h * 0.36, top, cx + head_h * 0.36, top + head_h * 0.98), fill=body)
    draw.rounded_rectangle(
        (cx - head_h * 0.19, top + head_h * 0.82, cx + head_h * 0.19, y_shoulder + 4), head_h * 0.1, fill=body
    )

    # 몸통: 어깨 -> 가슴 -> 허리 -> 엉덩이 -> 가랑이. 좌우 대칭 폴리곤.
    half = [
        (shoulder_w / 2, y_shoulder),
        (chest_w / 2, y_chest),
        (waist_w / 2, y_waist),
        (hip_w / 2, y_hip),
        (hip_w / 2 * 0.94, y_crotch),
    ]
    draw.polygon(
        [(cx + dx, y) for dx, y in half] + [(cx - dx, y) for dx, y in reversed(half)],
        fill=body,
    )

    # 팔: A-pose라 몸통에서 바깥으로 벌어진다. 어깨 관절에서 시작해 손목까지.
    # 측면 뷰에서는 한 팔만 보이고, 몸통 옆에 거의 붙어 내려온다.
    upper_arm_w = chest_w * 0.20
    for side in ((1,) if side_view else (-1, 1)):
        sx = cx + side * (shoulder_w / 2 - upper_arm_w * 0.30)
        elbow = (sx + side * upper_arm_w * 0.85, at(0.360))
        wrist = (elbow[0] + side * upper_arm_w * 0.55, at(0.470))
        draw.line([(sx, y_shoulder), elbow], fill=body, width=int(upper_arm_w))
        draw.line([elbow, wrist], fill=body, width=int(upper_arm_w * 0.78))
        draw.ellipse(
            (wrist[0] - upper_arm_w * 0.40, wrist[1] - upper_arm_w * 0.18,
             wrist[0] + upper_arm_w * 0.40, wrist[1] + upper_arm_w * 0.62),
            fill=body,
        )

    # 다리: 허벅지에서 발목까지 좁아지는 사다리꼴 + 발.
    # 측면에서는 두 다리가 거의 겹치므로 벌어짐을 없애고 발만 앞으로 낸다.
    thigh_w = hip_w * (0.86 if side_view else 0.46)
    ankle_w = thigh_w * 0.42
    gap = hip_w * 0.04
    y_ankle = at(0.962)
    leg_spread = 0.06 if side_view else 0.24
    for side in (-1, 1):
        hx = cx + side * hip_w * leg_spread
        ax = cx + side * hip_w * (leg_spread * 0.85)
        draw.polygon(
            [
                (hx - thigh_w / 2, y_hip), (hx + thigh_w / 2, y_hip),
                (ax + ankle_w / 2, y_ankle), (ax - ankle_w / 2, y_ankle),
            ],
            fill=body,
        )
        foot_back, foot_front = (0.5, 2.1) if side_view else (0.55, 0.75)
        draw.rounded_rectangle(
            (ax - ankle_w * foot_back, y_ankle - 2, ax + ankle_w * foot_front, ground), ankle_w * 0.3, fill=body
        )
    # 두 다리 사이를 가랑이 위쪽에서만 붙여 준다.
    draw.polygon(
        [(cx - gap, y_hip), (cx + gap, y_hip), (cx + gap * 1.4, y_crotch), (cx - gap * 1.4, y_crotch)],
        fill=body,
    )
    canvas = canvas.filter(ImageFilter.GaussianBlur(1.2))
    if view == "back":
        canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return canvas, goals


def build_body_reference(target: BodyTarget, view: str = DEFAULT_VIEW) -> BodyReference:
    """치수 -> FLUX에 넘길 체형 레퍼런스 + 달성 치수.

    SMPL-X 가중치가 있으면 메시를 해당 각도로 돌려 렌더하고, 없으면 비율
    실루엣으로 폴백한다. 어느 쪽이든 반환 타입은 같으므로 호출부는 분기하지
    않아도 된다. 달성 치수는 시점과 무관하므로 어느 뷰에서든 같은 값이 나온다.
    """
    if view not in VIEW_YAW:
        raise ValueError(f"unknown view: {view}")
    goals = target.goals()
    if SMPLX_MODEL_PATH and os.path.isdir(SMPLX_MODEL_PATH):
        try:
            betas, vertices, achieved = fit_betas(target)
            return BodyReference(
                image=render_mesh_silhouette(vertices, yaw=VIEW_YAW[view]),
                achieved=achieved,
                target={k: v for k, v in goals.items() if k in achieved},
                betas=[round(float(b), 3) for b in betas],
                source="smplx-fitted",
                view=view,
            )
        except Exception:  # 가중치 손상, smplx 미설치, 최적화 실패 등
            import logging

            logging.exception("SMPL-X fitting failed; falling back to the proportional silhouette")

    image, resolved = render_proportional_silhouette(target, view=view)
    return BodyReference(
        image=image, achieved={}, target=resolved, source="silhouette-fallback", view=view
    )


# 시점별로 달라지는 부분만 표에 두고 나머지는 공통 문장으로 묶는다.
# 문장 전체를 뷰마다 복사하면 한 곳만 고쳐도 뷰끼리 조명·복장이 어긋난다.
VIEW_DIRECTION = {
    "front": (
        "Front-facing relaxed symmetrical A-pose, arms slightly away from the torso, face toward the camera"
    ),
    "side": (
        "Exact left profile, the body turned 90 degrees so only one side faces the camera, "
        "arms hanging relaxed beside the torso, gaze straight ahead in profile"
    ),
    "back": (
        "Seen from directly behind, the back of the head, the hairline and the shoulder blades toward "
        "the camera, the face turned entirely away, arms slightly away from the torso"
    ),
}

# 발이 잘리는 문제는 CSS(object-fit)와 생성 프레이밍 양쪽에서 났다. 여기서는
# 후자를 막는다 — 머리 위와 발밑에 빈 여백을 명시적으로 요구한다.
FRAMING = (
    "Full body from the top of the head to the soles of the feet inside the frame, with clear empty "
    "background above the head and below the feet, both feet flat on the ground and completely visible, "
    "eye-level camera, 85 mm catalog lens"
)

STYLING = (
    "Wear a plain fitted charcoal crew-neck top and fitted mid-thigh charcoal shorts so the body outline "
    "stays readable. Realistic skin with natural Korean facial features, bare arms and legs, hair kept short "
    "and close to the head. Clean warm-grey seamless studio background, soft even lighting, single frame, "
    "clean image with no lettering."
)


def build_avatar_prompt(
    target: BodyTarget, view: str = DEFAULT_VIEW, identity_reference: bool = False
) -> str:
    """체형 가이드를 따라 그리라는 지시문.

    identity_reference=True 면 참조 이미지 2에 이미 만들어둔 정면 아바타가 들어
    있다는 뜻이다. 측면·후면을 체형 가이드만 보고 따로 생성하면 매번 다른 사람이
    나오므로, 완성된 정면을 함께 넘겨 "같은 사람을 돌린 것"이라고 못 박는다.
    """
    if view not in VIEW_DIRECTION:
        raise ValueError(f"unknown view: {view}")
    gender = "adult Korean woman" if target.gender == "women" else "adult Korean man"

    parts = [
        f"Reference image 1 is a body-shape guide: a plain grey mannequin form seen from the {view}, "
        "whose height and torso proportions are exact. Redraw this exact body as a photorealistic "
        f"{gender}, matching the guide's silhouette width line for line and keeping its head-to-body "
        "ratio and leg length."
    ]
    if identity_reference:
        parts.append(
            "Reference image 2 is the same person already photographed from the front. Keep that exact "
            "person — same face structure, hair, skin tone, build and clothing — and show them rotated "
            f"to the {view} view. This is the same photo session, so lighting and background match."
        )
    parts.append(f"{VIEW_DIRECTION[view]}. {FRAMING}.")
    parts.append(STYLING)
    return " ".join(parts)


# --- 전신사진 -> 스튜디오 아바타 ---------------------------------------------

PHOTO_AVATAR_PROMPT = (
    "Reference image 1 is a photograph of a real person. Redraw that same person as a clean full-length "
    "studio catalog photograph. Keep their identity exactly: the same face and facial features, the same "
    "hairstyle and hair colour, the same skin tone, the same body build and proportions, and the same "
    "clothes they are wearing in the photograph, reproduced in their real colours and cut. "
    "Place them standing upright and front-facing in a relaxed symmetrical A-pose with arms slightly away "
    "from the torso and both feet flat on the ground. "
    "Replace the original surroundings with a clean warm-grey seamless studio backdrop under soft even "
    "lighting, so the person stands alone in the frame. "
    "Full body from the top of the head to the soles of the shoes inside the frame, with clear empty "
    "background above the head and below the feet, eye-level camera, 85 mm catalog lens. "
    "Photorealistic single frame, clean image with no lettering."
)


def build_photo_avatar_prompt() -> str:
    """실제 사진을 스튜디오 아바타로 바꾸는 지시문.

    체형 가이드가 없다 — 체형 정보가 이미 사진 안에 있고, 마네킹을 끼워 넣으면
    오히려 그 인물의 실제 몸을 밀어낸다. 사진의 배경과 포즈만 갈아끼우고
    사람은 그대로 두는 편집이다.
    """
    return PHOTO_AVATAR_PROMPT
