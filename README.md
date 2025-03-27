# Retrieval Augmented Generation with a GGUF Model

This repository shows how to set up a RAG pipeline using a custom GGUF model, downloaded from HuggingFace
Along with LangChain, Chroma for vector storage and Streamlit for a chatbot interface. 

In building this, I have closely followed the structure [here](https://medium.com/@vipra_singh/building-llm-applications-open-source-chatbots-part-7-1ca9c3653175) (the repo in section 5) but I am using a different model, and some parts of my script are different. This is just to show that the original structure can be easily extended. 

 **Note:** This project is a demonstration. Some components (such as ingestion performance and retrieval accuracy) may require 
 further optimization.

 ## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Ingesting Documents](#ingesting-documents)
  - [Querying the Pipeline](#querying-the-pipeline)
  - [Chatbot Interface](#chatbot-interface)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [License](#license)

- ## Features

- **Custom GGUF Model Integration:**  
  Uses a GGUF model (from [Hugging Face](https://huggingface.co/)) wrapped as a custom LLM for generation.

- **Retrieval-Augmented Generation (RAG):**  
  Combines a vector store (Chroma), a retrieval module, and a language model to answer questions based on ingested documents.

- **Document Ingestion:**  
  Extracts text from PDFs or plain text files, splits the text into manageable chunks, and indexes them in a vector store for retrieval.

- **Chatbot Interface:**  
  A simple Streamlit-based interface for querying the RAG pipeline interactively.

---


## Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- A **GGUF model file** I have downloaded [`mistral-7b-instruct-v0.2.Q2_K.gguf`](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q2_K.gguf). This is a [quantised model](https://medium.com/@florian_algo/model-quantization-1-basic-concepts-860547ec6aa9#:~:text=In%20mathematics%20and%20digital%20signal,in%20the%20field%20of%20algorithms.) not very space consuming (3.08 GB) and compatible with llama.cpp and llama-cpp-python. If you have enough space you can use better models or, alternatively, use Ollama and some of the LLMs models platformed there. But if you want to test this repository, you must have my model installed on your machine.
- Hardware requirements based on your model (CPU is supported; GPU or Apple Metal can improve performance) 

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gianludam/AI_Project.git
   cd AI_Project
