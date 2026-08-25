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
    print(f"Notebook valid: {len(notebook['cells'])} cells, {code_cells} Python cells")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} NOTEBOOK.ipynb")
    raise SystemExit(validate(Path(sys.argv[1])))
