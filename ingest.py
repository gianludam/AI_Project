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

def split_text(doc_text, chunk_size, chunk_overlap):
    """
    Splits document text into manageable chunks using RecursiveCharacterTextSplitter.
    
    Args:
        doc_text (str): The document text to be split.
        chunk_size (int): The target size (in characters) for each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        
    Returns:
        list[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_text(doc_text)
    texts = text_splitter.create_documents(splits)
    print(f"Loaded {len(texts)} splits")
    # Each element in 'texts' is a Document object. We extract just the text.
    return texts

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
    for i, chunk in enumerate(chunks):
        vectorstore.add_documents([chunk], ids=[f"doc_{i}"])
    
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

