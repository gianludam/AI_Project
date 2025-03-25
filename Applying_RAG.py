import json
import timeit
import streamlit as st
import box
import yaml
import nltk
nltk.download('punkt')

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from llama_cpp import Llama

#########################################
# Custom LLM class wrapping your GGUF model
#########################################
class CustomGGUFXMLLM(LLM):
    __private_attributes__ = {"_llm": None}  # Define a private attribute for the model
    
    """
    Custom LLM class wrapping your GGUF model.
    Ensures the model follows the LangChain LLM interface.
    """
    def __init__(self, model_path, n_ctx, n_batch, max_tokens=512, temperature=0.7, top_p=0.9, **kwargs):
        """
        Initialize your GGUF model using llama_cpp's Llama.
        
        Args:
            model_path (str): File path to your GGUF model.
            n_ctx (int): Context length (number of tokens the model can process).
            n_batch (int): Batch size for processing inputs.
            max_tokens (int): Maximum tokens to generate in a response.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls the nucleus sampling cutoff.
            **kwargs: Additional keyword arguments for the model.
        """
        object.__setattr__(self, "_llm", Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            **kwargs
        ))
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    @property
    def _llm_type(self) -> str:
        """
        Return the type of LLM. This helps LangChain identify the model.
        """
        return "custom-gguf"

    def _call(self, prompt: str, stop=None) -> str:
        """
        Process the prompt and return the generated text.
        
        Args:
            prompt (str): The input prompt.
            stop (optional): Stop tokens for generation.
            
        Returns:
            str: The generated response text.
        """
        output = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
            echo=False
        )
        return output["choices"][0]["text"]

#########################################
# Embedding model loader function
#########################################
def load_embedding_model(model_name: str, device: str = "cpu", normalize: bool = True):
    """
    Create and return an instance of HuggingFaceEmbeddings.
    
    Args:
        model_name (str): The name or identifier of the embedding model you want to load 
                          (e.g., "multi-qa-MiniLM-L6-cos-v1").
        device (str): The device on which to run the model (default "cpu").
        normalize (bool): Whether to normalize embeddings (default True).
    
    Returns:
        HuggingFaceEmbeddings: The loaded embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize}
    )

#########################################
# Retriever loader function using Chroma DB
#########################################
def load_retriever(embedding_model, persist_directory, collection_name, vector_space="cosine", top_k=3):
    """
    Create a Chroma vector store and return a retriever for similarity search.
    
    Args:
        embedding_model: The embedding model to encode text.
        persist_directory (str): Directory where the vector store persists data.
        collection_name (str): Name of the collection in the vector store.
        vector_space (str): The metric space used (default "cosine").
        top_k (int): Number of top documents to retrieve (default 3).
    
    Returns:
        Retriever: A retriever object for similarity search.
    """
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": vector_space},
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever

#########################################
# Prompt template loader function
#########################################
def load_prompt_template():
    """
    Load and return a prompt template for guiding the LLM.
    
    Returns:
        PromptTemplate: The prompt template.
    """
    template = """Use the following context to answer the question. 
If you don't know the answer, say "I don't know."

Context: {context}
Question: {question}

Answer:"""
    return PromptTemplate.from_template(template)

#########################################
# QA chain loader function
#########################################
def load_qa_chain(llm, retriever, prompt):
    """
    Create a RetrievalQA chain that combines the LLM, retriever, and prompt template.
    
    Args:
        llm: The language model.
        retriever: The retriever for fetching relevant documents.
        prompt: The prompt template.
    
    Returns:
        RetrievalQA: The configured QA chain.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

#########################################
# Build the full RAG pipeline
#########################################
def build_rag_pipeline(config_path="config.yml"):
    """
    Build and return the complete RAG pipeline by loading configuration,
    embedding model, vector store, retriever, prompt, and QA chain.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        RetrievalQA: The configured QA chain.
    """
    with open(config_path, 'r') as ymfile:
        cfg = box.Box(yaml.safe_load(ymfile))

    print("Loading embedding model...")
    embeddings = load_embedding_model(
        model_name=cfg.EMBEDDINGS,
        device=cfg.DEVICE,
        normalize=cfg.NORMALIZE_EMBEDDINGS
    )

    print("Loading vector store and retriever...")
    retriever = load_retriever(
        embedding_model=embeddings,
        persist_directory=cfg.VECTOR_DB,
        collection_name=cfg.COLLECTION_NAME,
        vector_space=cfg.VECTOR_SPACE,
        top_k=cfg.NUM_RESULTS
    )

    print("Loading prompt template...")
    prompt = load_prompt_template()

    print("Loading Model...")
    llm = CustomGGUFXMLLM(
        model_path=cfg.GGUF_MODEL_PATH,
        n_ctx=cfg.N_CTX,
        n_batch=cfg.N_BATCH,
        max_tokens=cfg.MAX_TOKENS,
        temperature=cfg.TEMPERATURE,
        top_p=cfg.TOP_P
    )

    print("Loading QA chain...")
    qa_chain = load_qa_chain(llm, retriever, prompt)
    return qa_chain



 






