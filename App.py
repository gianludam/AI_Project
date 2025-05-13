import box
from box import Box
import yaml
import PyPDF2

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pathlib import Path
from typing import Iterable
import io
import pandas as pd

from query_script import get_rag_response
import timeit
from htmlTemplates import css
from ingest import ingest_documents

@st.cache_resource(show_spinner="⌛ Loading model & vector‑store…")
def get_qa_chain():
    from Applying_RAG import build_rag_pipeline
    return build_rag_pipeline() 

        
def extract_text(file_paths: Iterable[Path]) -> str:
    out = []
    for fp in file_paths:
        suffix = fp.suffix.lower()
        try:
            if suffix == ".pdf":
                # … existing PDF code …
            elif suffix in {".txt",".md",".csv",".json",".html",".htm"}:
                # … existing plain‑text code …
            elif suffix == ".docx":
                # … existing DOCX code …
            elif suffix == ".epub":
                # … existing EPUB code …
            elif suffix in {".png",".jpg",".jpeg"}:
                # … existing OCR code …
            # ─────────────────  NEW  ─────────────────
            elif suffix in {".xlsx", ".xls"}:
                df_dict = pd.read_excel(fp, sheet_name=None)
                for sheet, df in df_dict.items():
                    out.append(f"### Sheet: {sheet} — {len(df)} rows × {len(df.columns)} cols")
                    out.extend(df.astype(str).fillna("").agg(" | ".join, axis=1))
            # ─────────────────────────────────────────
            else:
                st.warning(f"Unsupported file skipped: {fp.name}")
        except Exception as e:
            st.warning(f"Could not extract {fp.name}: {e}")
    return "\n".join(out)


def handle_userinput(user_question: str) -> None:
    """
    Runs the cached RAG pipeline on the user question and streams the answer.
    """
    qa_chain = get_qa_chain()            # ← comes from the cache above

    start = timeit.default_timer()
    answer = get_rag_response(user_question, qa_chain)
    end   = timeit.default_timer()

    st.write(answer)
    st.markdown(f"**Time to retrieve answer:** {end - start:.2f} s")

def main() -> None:
    """
    The main function of the Streamlit application.

    Loads environment variables, configures the Streamlit app layout,
    handles user interaction, and processes PDFs.
    """
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)  # Apply CSS styles

    # Load configuration for ingestion (assuming config.yml is in the same directory)
    with open("config.yml", "r") as f:
        cfg = box.Box(yaml.safe_load(f))

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :books:")
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

   with st.sidebar:
    st.subheader("Documents & datasets")

    docs = st.file_uploader(
        "Upload files and click **Ingest**",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "xls", "csv", "txt", "md"],
    )

    if st.button("Ingest"):
        if docs:
            with st.spinner("Processing"):
                # a throw‑away folder that auto‑deletes afterwards
                with TemporaryDirectory() as tmpdir:
                    paths = []

                    for up in docs:
                        p = Path(tmpdir) / up.name
                        p.write_bytes(up.getbuffer())
                        paths.append(p)

                    text = extract_text(paths)      # 🔹 NEW universal extractor
                    ingest_documents(text, cfg)     # your existing splitter+embedder

            st.success("✅ Ingestion complete!")
        else:
            st.warning("📂 Please upload at least one document.")

if __name__ == '__main__':
    main()
