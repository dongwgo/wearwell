from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(path: Path) -> int:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Not a valid nbformat 4 notebook")

    code_cells = 0
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = "".join(cell.get("source", []))
        compile(source, f"{path.name}:cell-{index}", "exec")

    if code_cells == 0:
        raise ValueError("Notebook has no code cells")
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = (
        "files.download",
        "local-config.js",
        "BACKEND_URL=",
        "Qwen/Qwen3-VL-8B-Instruct",
        '\"VLM_LOAD_IN_4BIT\": \"1\"',
        'health.get(\"vlmWarmupVerified\")',
    )
    forbidden = ("APP_URL=", 'REPO_DIR / "config.js"', "인증된 Wearwell 열기")
    if missing := [value for value in required if value not in all_source]:
        raise ValueError(f"Notebook is missing API-only setup: {missing}")
    if present := [value for value in forbidden if value in all_source]:
        raise ValueError(f"Notebook still hosts/configures the frontend: {present}")
    print(f"Notebook valid: {len(notebook['cells'])} cells, {code_cells} Python cells")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} NOTEBOOK.ipynb")
    raise SystemExit(validate(Path(sys.argv[1])))
