"""Vistas interactivas del experimento y del asistente RAG."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from umsa.analytics import DatasetCatalog
from umsa.lessons import experiment_dashboard
from umsa.notebook import notebook_cells
from umsa.rag import RAGError, RAGIndex, chat_with_data, knowledge_chunks, load_settings


@st.cache_data(show_spinner=False)
def get_notebook():
    return notebook_cells()


@st.cache_resource(show_spinner="Preparando el índice local de las tesis y del experimento…")
def get_rag_index():
    """Índice documental con los 79 metadatos, documentación vigente y corpus completo."""
    from umsa.data import load_benchmark, load_corpus

    return RAGIndex(knowledge_chunks(load_corpus(), load_benchmark()))


@st.cache_resource(show_spinner="Leyendo los datasets del proyecto…")
def get_dataset_catalog():
    """Catálogo enriquecido con fichas, materias y trazabilidad de exclusiones."""
    return DatasetCatalog()


@st.cache_data(show_spinner=False)
def conclusion_similarity(corpus: pd.DataFrame):
    texts = corpus["conclusiones_limpias"].fillna("").astype(str)
    matrix = TfidfVectorizer(max_features=1500, min_df=2).fit_transform(texts)
    return cosine_similarity(matrix)


def experiment_page(corpus: pd.DataFrame, metadata: pd.DataFrame, chart):
    experiment_dashboard(corpus, metadata, chart)


def render_sources(sources):
    with st.expander(f"Consultar evidencia · {len(sources)} fragmentos"):
        for source in sources:
            st.markdown(f"**[{source['id']}] {source['title']}**")
            if source.get("kind") == "consulta":
                st.caption(source["source"])
                st.dataframe(pd.DataFrame(source["rows"]), hide_index=True, width="stretch")
                st.code(source["sql"], language="sql")
                continue
            if source.get("kind") == "ficha":
                st.caption(source["source"])
                rows = pd.DataFrame(source["rows"])
                visible = [col for col in ("archivo_pdf", "titulo", "autores", "fecha", "materias", "uri") if col in rows]
                st.dataframe(rows[visible], hide_index=True, width="stretch")
                st.json(json.loads(source["text"]), expanded=False)
                continue
            if source.get("kind") == "pagina":
                st.caption(f"{source['source']} de {source['total_pages']} · carácter {source['offset']}")
                st.text(source["text"])
                continue
            st.caption(f"{source['source']} · carácter {source['offset']} · "
                       f"afinidad léxica {source['score']:.3f}")
            st.text(source["text"])


def chat_page(corpus: pd.DataFrame):
    st.html('<div class="eyebrow">ASISTENTE / EVIDENCIA RECUPERADA</div>')
    st.title("Conversa con el experimento")
    st.write("Consulta docentes, rankings, años y estadísticas de los datasets, o pregunta por la metodología "
             "y el contenido de las tesis. Puedes revisar los datos y documentos usados en cada respuesta.")
    settings = load_settings()
    if not settings.api_key:
        st.info("Configura OPENROUTER_API_KEY en .env o en el entorno del servidor para activar GPT-4o mini. "
                "La búsqueda de evidencia sigue disponible sin una clave.")
    with st.container(border=True):
        left, middle, right = st.columns([1, 1.5, 1])
        scope_label = left.selectbox("Buscar en", ["Todo el proyecto", "Sólo experimento", "Una tesis"])
        scope = {"Todo el proyecto": "Todo", "Sólo experimento": "experimento", "Una tesis": "tesis"}[scope_label]
        filename = ""
        if scope_label == "Una tesis":
            projects = get_dataset_catalog().tables["proyectos"]
            labels = dict(zip(projects["archivo_pdf"], projects["titulo"]))
            filename = middle.selectbox("Documento", list(labels),
                                        format_func=lambda f: f"{f} · {labels[f][:55]}")
        else:
            middle.caption("Fuentes: todos los CSV y Parquet, metadatos originales, cuaderno y texto de las tesis.")
        semantic = right.toggle("Reordenar con embeddings", value=bool(settings.embedding_key),
                                disabled=not bool(settings.embedding_key),
                                help="Reordena hasta 24 candidatos locales con text-embedding-3-small en OpenRouter.")
        preview = st.toggle("Sólo buscar evidencia", value=not bool(settings.api_key))
        st.caption(f"Modelo: {settings.model} · Las preguntas y los fragmentos seleccionados se envían a OpenRouter al consultar.")
    messages = st.session_state.setdefault("rag_messages", [])
    if st.button("Nueva conversación", disabled=not messages):
        st.session_state["rag_messages"] = []
        st.rerun()
    if not messages:
        st.markdown("**Puedes empezar con una de estas preguntas**")
        suggestions = ["¿Cuáles son los docentes con más proyectos?",
                       "¿Qué resultados obtuvo Random Forest en el benchmark?",
                       "Busca proyectos sobre inventarios y dime sus autores.",
                       "¿Qué objetivos y conclusiones tiene el proyecto de Seshirya?",
                       "¿Cómo se calculó el índice de rigor?",
                       "¿Qué proyectos quedaron fuera del corpus final?"]
        suggestion = None
        for start in (0, 3):
            for col, text in zip(st.columns(3), suggestions[start:start + 3]):
                if col.button(text, width="stretch"):
                    suggestion = text
    else:
        suggestion = None
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            if message.get("mode"):
                st.caption(message["mode"])
    prompt = st.chat_input("Escribe una pregunta sobre el proyecto…", max_chars=3000) or suggestion
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.spinner("Buscando evidencia y preparando la respuesta…"):
                if preview:
                    index = get_rag_index()
                    sources, mode = index.retrieve(prompt, scope, filename, settings=settings, semantic=semantic)
                    response = {"answer": "Estos son los fragmentos encontrados para tu pregunta." if sources
                                else "No encontré evidencia en el ámbito seleccionado. Prueba ampliar la búsqueda.",
                                "sources": sources, "usage": {}}
                else:
                    response = chat_with_data(prompt, get_rag_index, get_dataset_catalog(), settings, messages,
                                              scope=scope, filename=filename, semantic=semantic)
                    mode = response["mode"]
            usage = response.get("usage", {}).get("total_tokens")
            details = f"Recuperación: {mode}" + (f" · {usage} tokens" if usage else "")
            messages.extend([{"role": "user", "content": prompt},
                             {"role": "assistant", "content": response["answer"],
                              "sources": response["sources"], "mode": details}])
            st.rerun()
        except RAGError as exc:
            st.error(str(exc))
            st.caption("Tu consulta no se añadió al historial. Puedes volver a intentarlo.")
    if messages:
        st.download_button("Descargar conversación y fuentes", json.dumps(messages, ensure_ascii=False, indent=2),
                           "conversacion_umsa.json", "application/json")
