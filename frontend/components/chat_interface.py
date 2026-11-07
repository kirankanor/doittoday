"""Chat history, message rendering, and citations."""
import streamlit as st
from services.backend_client import ask_backend, BackendError


def render_sources(citations: list[dict]) -> None:
    """Show source document, page number, chunk ID, and optional excerpt."""
    if not citations:
        st.caption("No source references returned.")
        return
    with st.expander(f"Sources ({len(citations)})"):
        for i, source in enumerate(citations, start=1):
            name = source.get("document_name") or source.get("source") or "Unknown document"
            page = source.get("page_number", "Not provided")
            chunk_id = source.get("chunk_id")
            st.markdown(f"**{i}. {name}**")
            st.caption(f"Page {page}" + (f" · {chunk_id}" if chunk_id else ""))
            if source.get("text"):
                st.write(source["text"])
            if i < len(citations):
                st.divider()


def render_chat_interface() -> None:
    """Render conversation history and accept a new question."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Streamlit reruns the script after interactions; replay saved messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("citations", []))

    question = st.chat_input("Ask a question about your internal documents...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching internal documents..."):
            try:
                result = ask_backend(question)
                answer = result["answer"]
                citations = result.get("citations", [])
                st.markdown(answer)
                render_sources(citations)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "citations": citations
                })
            except BackendError as exc:
                error = f"Could not get an answer: {exc}"
                st.error(error)
                st.session_state.messages.append({
                    "role": "assistant", "content": error, "citations": []
                })
