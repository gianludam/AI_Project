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

class CustomGGUFXMLLM(LLM):
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
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            **kwargs
        )
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
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
            echo=False
        )
        return output["choices"][0]["text"]


 






