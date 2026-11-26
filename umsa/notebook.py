"""Lectura de evidencia guardada del cuaderno, sin ejecutar código de Colab."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "repositorioUMSA.ipynb"


def notebook_cells(path: Path = NOTEBOOK_PATH) -> list[dict]:
    if not path.exists():
        return []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        outputs, images = [], []
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            text = "".join(output.get("text", data.get("text/plain", [])))
            text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
            if text.strip() and output.get("name") != "stderr":
                outputs.append(text.strip())
            if "image/png" in data:
                images.append(base64.b64decode("".join(data["image/png"])))
        meaningful = next((line.lstrip("# ") for line in source.splitlines() if line.startswith("# ")), "")
        result.append({
            "cell": index + 1,
            "kind": cell.get("cell_type", "code"),
            "title": meaningful or f"Celda {index + 1}",
            "source": source,
            "output": "\n\n".join(outputs),
            "images": images,
        })
    return result


def chunk_text(text: str, size: int = 1800, overlap: int = 250) -> list[tuple[int, str]]:
    """Fragmenta por caracteres con solapamiento, preservando localizadores."""
    if size <= overlap or overlap < 0:
        raise ValueError("El tamaño debe superar el solapamiento.")
    text = str(text).strip()
    return [(start, text[start:start + size]) for start in range(0, len(text), size - overlap)]
