import os
import argparse
import timeit
import box
import yaml
import warnings

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import functools

from embedding_cache import get_embedder

def split_text(doc_text, chunk_size, chunk_overlap, source_name="uploaded_file"):
    """
    Splits a long document text into smaller overlapping chunks
    and wraps each chunk in a LangChain Document object with metadata.

    Args:
        doc_text (str): The full text content of the document to be split.
        chunk_size (int): The maximum number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        source_name (str): A label identifying the source of the text, 
                           useful for debugging or displaying document provenance.

    Returns:
        list[Document]: A list of LangChain Document objects, each containing:
            - page_content (str): The text content of a chunk
            - metadata (dict): Metadata including the 'source' name
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Directly create Document objects with metadata
    documents = text_splitter.create_documents(
        texts=[doc_text],
        metadatas=[{"source": source_name}]
    )

    print(f"Loaded {len(documents)} chunks")
    return documents

def ingest_documents(doc_text, cfg):
    """
    Ingests document text into the Chroma vector store.
    
    Args:
        doc_text (str): The document text to ingest.
        cfg: Configuration object containing keys from config.yml.
    """
    # Load the embedding model using HuggingFaceEmbeddings
    embeddings =  get_embedder(cfg.EMBEDDINGS, cfg.DEVICE, cfg.NORMALIZE_EMBEDDINGS) 

    # Initialize or load the Chroma vector store
    vectorstore = Chroma(
        persist_directory=cfg.VECTOR_DB,
        collection_name=cfg.COLLECTION_NAME,
        collection_metadata={"hnsw:space": cfg.VECTOR_SPACE},
        embedding_function=embeddings
    )
    print(f"Vector store created at {cfg.VECTOR_DB}")
  
    # Split the document text into chunks
    chunks = split_text(doc_text, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    print(f"Total chunks created: {len(chunks)}")
    
    # Add each chunk to the vector store with a unique ID
    vectorstore.add_documents(chunks)

    
    # Persist the vector store to disk
    vectorstore.persist()
    print("Ingestion complete: All chunks have been added to the vector store.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_file', type=str, required=True,
                        help="Path to a text file containing the document text to ingest.")
    parser.add_argument('--config', type=str, default="config.yml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load configuration from YAML using box for dot-notation
    with open(args.config, 'r') as f:
        cfg = box.Box(yaml.safe_load(f))

    # Read document text from the provided file path
    with open(args.doc_file, 'r', encoding='utf-8') as file:
        doc_text = file.read()

    print(f"Ingesting document from: {args.doc_file}")

    start = timeit.default_timer()
    ingest_documents(doc_text, cfg)
    end = timeit.default_timer()
    
    print(f"Time taken for ingestion: {end - start:.2f} seconds")

