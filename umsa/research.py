"""Búsqueda de proyectos y lectura dirigida de evidencia con localizadores reales."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer

from umsa.analytics import DatasetCatalog, DatasetQueryError, normalized_text
from umsa.notebook import ROOT, chunk_text, notebook_cells

SEARCH_STOPWORDS = {"el", "la", "los", "las", "de", "del", "en", "con", "y", "o", "por", "para", "que",
                    "cual", "cuales", "un", "una", "proyectos", "proyecto", "tesis", "sobre", "busca", "buscar",
                    "dame", "hay", "relacionados", "relacionadas", "me", "muestra", "mostrar"}


def terms(query: str) -> list[str]:
    return [word for word in re.findall(r"\w+", normalized_text(query)) if len(word) >= 2 and word not in SEARCH_STOPWORDS]


def excerpt(text: str, query: str, size: int = 2400) -> tuple[int, str]:
    words = terms(query)
    normalized = normalized_text(text)
    # El texto normalizado cambia espacios: los localizadores siempre se refieren al original.
    positions = [match.start() for word in words for match in re.finditer(re.escape(word), text.lower())]
    start = max(0, min(positions, default=0) - 200)
    if normalized and not positions:
        # Busca por líneas para conservar offsets aun cuando se eliminan tildes.
        offset = 0
        for line in text.splitlines(keepends=True):
            if any(word in normalized_text(line) for word in words):
                start = max(0, offset - 200)
                break
            offset += len(line)
    return start, text[start:start + size]


@lru_cache(maxsize=8)
def pdf_pages(path: str, modified: int) -> tuple[str, ...]:
    reader = PdfReader(path)
    return tuple(page.extract_text() or "" for page in reader.pages)


class ProjectResearch:
    def __init__(self, catalog: DatasetCatalog):
        self.catalog = catalog

    def search(self, query: str = "", docente: str = "", autor: str = "", anio: int | None = None,
               limit: int = 10, offset: int = 0, filename: str = "") -> dict:
        if not all(isinstance(v, str) for v in (query, docente, autor)):
            raise DatasetQueryError("Los filtros de búsqueda deben ser texto.")
        limit, offset = max(1, min(20, int(limit))), max(0, int(offset))
        frame = self.catalog.tables["proyectos"]
        if filename:
            frame = frame.loc[frame["archivo_pdf"].eq(filename)]
        if anio is not None:
            frame = frame.loc[frame["fecha"].eq(int(anio))]
        for field, value in (("tutores", docente), ("autores", autor)):
            if value.strip():
                needles = terms(value)
                frame = frame.loc[frame[field].fillna("").map(lambda text, ns=needles: all(
                    needle in normalized_text(text) for needle in ns))]
        # Flexión plural simple para que «inventarios» también encuentre «inventario».
        words = [word[:-2] if word.endswith("es") and len(word) > 4
                 else word[:-1] if word.endswith("s") and len(word) > 4 else word for word in terms(query)]
        ranked = []
        for _, row in frame.iterrows():
            title = normalized_text(str(row["titulo"]))
            details = normalized_text(" ".join(str(row.get(col, "")) for col in ("materias", "resumen", "autores", "archivo_pdf")))
            score = sum(4 * int(word in title) + int(word in details) for word in words)
            if words and not score:
                continue
            ranked.append((score, str(row["archivo_pdf"]), row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        rows = []
        for score, _, row in ranked[offset:offset + limit]:
            record = {field: row.get(field) for field in
                      ("archivo_pdf", "titulo", "autores", "tutores", "fecha", "materias", "uri", "en_corpus_final", "pdf_disponible")}
            record["relevancia"] = score
            rows.append(json.loads(pd.DataFrame([record]).to_json(orient="records", force_ascii=False))[0])
        return {"rows": rows, "total_matches": len(ranked), "offset": offset,
                "next_offset": offset + limit if offset + limit < len(ranked) else None,
                "criteria": {"texto": query, "docente": docente, "autor": autor, "anio": anio},
                "note": "Búsqueda por coincidencias en título, materias y resumen; no clasificación semántica exhaustiva.",
                "files": [self.catalog.origins["proyectos"]]}

    def profile(self, filename: str, selected_filename: str = "") -> dict:
        record = self.catalog.project(filename, selected_filename)
        for key in ("resumen", "conclusiones_texto"):
            text = record.get(key)
            if isinstance(text, str) and len(text) > 4500:
                record[key] = text[:4500]
                record[key + "_truncado"] = True
        teachers = self.catalog.tables["docentes_original"]
        people = teachers.loc[teachers["archivo_pdf"].eq(filename), ["docente", "rol", "nombre_original"]]
        return {"project": record, "teachers": people.to_dict("records"),
                "files": [self.catalog.origins["proyectos"]],
                "note": "Los campos null son datos ausentes, no valores cero. Las métricas sólo existen para el corpus final."}

    def read_pdf(self, filename: str, query: str = "", page: int | None = None, selected_filename: str = "") -> list[dict]:
        if not isinstance(query, str):
            raise DatasetQueryError("La consulta de lectura debe ser texto.")
        self.catalog.project(filename, selected_filename)
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise DatasetQueryError("Archivo PDF no válido.")
        path = self.catalog.directory / "PDFs" / filename
        if not path.is_file():
            raise DatasetQueryError("El PDF no está disponible localmente; consulta su ficha y los textos extraídos de los datasets.")
        try:
            pages = pdf_pages(str(path), path.stat().st_mtime_ns)
        except (PdfReadError, OSError, ValueError) as exc:
            raise DatasetQueryError(f"No se pudo extraer el PDF ({type(exc).__name__}). Usa las fuentes tabulares disponibles.") from None
        if page is not None:
            if not 1 <= int(page) <= len(pages):
                raise DatasetQueryError(f"Página fuera de rango: el archivo tiene {len(pages)} páginas físicas.")
            chosen = [int(page) - 1]
        elif query.strip():
            searchable = [normalized_text(text) for text in pages]
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=18000)
                matrix = vectorizer.fit_transform(searchable)
                scores = (matrix @ vectorizer.transform([normalized_text(query)]).T).toarray().ravel()
            except ValueError:
                raise DatasetQueryError("El PDF no contiene texto extraíble; puede requerir OCR.") from None
            headings = [pattern for word, pattern in (
                ("objetiv", r"^\s*(?:\d[\d.]*\s*)?objetivo(?:s| general)?\b"),
                ("conclusion", r"^\s*(?:\d[\d.]*\s*)?conclusiones\b"),
                ("metodol", r"^\s*(?:\d[\d.]*\s*)?metodologia\b"),
            ) if word in normalized_text(query)]
            section_pages = []
            for i, text in enumerate(pages):
                if len(re.findall(r"\.{4,}", text)) >= 3:
                    scores[i] *= .15  # El índice suele repetir los encabezados sin contener las secciones.
                elif any(re.search(pattern, normalized_text(line)) for pattern in headings for line in text.splitlines()):
                    scores[i] += .15
            for pattern in headings:
                matching = [i for i, text in enumerate(pages) if len(re.findall(r"\.{4,}", text)) < 3
                            and any(re.search(pattern, normalized_text(line)) for line in text.splitlines())]
                if matching:
                    section_pages.append(max(matching, key=lambda i: scores[i]))
            chosen = list(dict.fromkeys(section_pages + [int(i) for i in np.argsort(-scores) if scores[i] > 0]))[:3]
        else:
            chosen = [0]
        result = []
        for i in chosen:
            offset, text = excerpt(pages[i], query)
            result.append({"title": f"{filename} · página {i + 1}", "source": f"dataset/PDFs/{filename} · página física {i + 1}",
                           "filename": filename, "page": i + 1, "total_pages": len(pages), "offset": offset,
                           "text": text, "kind": "pagina", "score": 1.0})
        return result

    def documentation(self, query: str) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise DatasetQueryError("Indica qué deseas consultar de la documentación.")
        path = ROOT / "README.md"
        fragments = chunk_text(path.read_text(encoding="utf-8"))
        vectorizer = TfidfVectorizer(strip_accents="unicode", sublinear_tf=True)
        matrix = vectorizer.fit_transform([text for _, text in fragments])
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        files = [name for name in ("app.py", "compose.yaml", "Dockerfile", "pyproject.toml", ".env.example")
                 if (ROOT / name).is_file()]
        return [{"title": "Guía de uso de la aplicación", "source": "README.md", "kind": "experimento",
                 "text": text, "offset": offset, "score": float(scores[i]), "project_files": files}
                for i in np.argsort(-scores)[:3] for offset, text in [fragments[i]]]

    def read_notebook(self, cell: int, offset: int = 0) -> dict:
        cells = notebook_cells()
        if not 1 <= int(cell) <= len(cells):
            raise DatasetQueryError(f"El cuaderno tiene {len(cells)} celdas, numeradas desde 1.")
        selected = cells[int(cell) - 1]
        text = f"Código:\n{selected['source']}\n\nSalidas guardadas:\n{selected['output']}"
        offset = max(0, int(offset))
        return {"title": selected["title"], "source": f"repositorioUMSA.ipynb · celda {cell}",
                "text": text[offset:offset + 6500], "offset": offset,
                "next_offset": offset + 6500 if offset + 6500 < len(text) else None,
                "kind": "experimento", "score": 1.0}
