from __future__ import annotations
from tempfile import TemporaryDirectory
from typing import Iterable, List
import timeit
import box
import yaml

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pathlib import Path
import io
import pandas as pd

from query_script import get_rag_response
from htmlTemplates import css
from ingest import ingest_documents

st.markdown("### ✅ This is the updated version of App.py")

@st.cache_data
def load_cfg(path: str = "config.yml") -> Box:
    """Parse YAML only once per session."""
    with open(path, "r") as fh:
        return box.Box(yaml.safe_load(fh))

@st.cache_resource(show_spinner="⌛ Loading model & vector‑store…")
def get_qa_chain():
    from Applying_RAG import build_rag_pipeline
    return build_rag_pipeline() 



def extract_text(file_paths: Iterable[Path]) -> str:
    """
    Turn any file into plain text
    Return a single long string containing text from all files
    """
    out: list[str] = []
    for fp in file_paths:
        suffix = fp.suffix.lower()
        try:
            if suffix == ".pdf":
               reader = PdfReader(fp)
               for page in reader.pages:
                   out.append(page.extract_text() or "")
                   
            elif suffix in {".txt",".md",".csv",".json",".html",".htm"}:
                """
                Plain text, markdown, CSV, JSON, HTML...
                """
                out.append(fp.read_text(errors="ignore"))
                
            elif suffix in {".xlsx", ".xls", ".xlsm"}:
                """
                Excel (all sheets)
                """
                df_dict = pd.read_excel(fp, sheet_name=None)  # sheet → DataFrame
                for sheet, df in df_dict.items():
                    out.append(f"### Sheet: {sheet}  —  {len(df)} rows × {len(df.columns)} cols")
                    # join each row into a pipe‑separated string
                    out.extend(df.astype(str).fillna("").agg(" | ".join, axis=1))
                    out.append("")  # blank line separator

            # ▸  unsupported
            else:
                st.warning(f"Unsupported file skipped: {fp.name}")

        except Exception as exc:
            st.warning(f"Could not extract {fp.name}: {exc}")

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
    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)  # Apply CSS styles

    # Load configuration for ingestion
    cfg = load_cfg()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Documents :books:")
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Documents & datasets")

        uploads = st.file_uploader(
            "Upload files then click **Ingest**",
            accept_multiple_files=True,
            type=[
                "pdf", "txt", "md", "csv", "json", "html", "htm",
                "xlsx", "xls", "xlsm",
            ],
        )

        if st.button("Ingest"):
            if not uploads:
                st.warning("📂 Please upload at least one document.")
                st.stop()

            with st.spinner("🔄 Extracting & indexing…"):
                with TemporaryDirectory() as tmpdir:
                    paths: List[Path] = []
                    for uf in uploads:
                        p = Path(tmpdir) / uf.name
                        p.write_bytes(uf.getbuffer())
                        paths.append(p)

                    text_blob = extract_text(paths)
                    ingest_documents(text_blob, cfg)

            st.success("✅ Ingestion complete!")

# ──────────────────────────── Run ───────────────────────────
if __name__ == "__main__":
    main()
