"""
Configuration settings for the RAG system.
Contains all model names, file paths, and retrieval parameters.
"""

# --- MODEL CONFIGURATION ---
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'Mistral-7B-Instruct-v0.2-Q4_K_M:latest'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- FILE PATHS ---
CHROMA_DB_PATH = "./my_rag_db"
DATASET_PATH = "../assets/dataset.txt"

# --- RETRIEVAL PARAMETERS ---
TOP_N = 3
CANDIDATES_TO_RETRIEVE = 10

# --- COLLECTION SETTINGS ---
COLLECTION_NAME = "my_presentation_docs"
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# --- PROMPT TEMPLATE ---
INSTRUCTION_PROMPT = """You are an expert Q&A assistant. Your task is to directly answer the user's question based *only* on the provided context. Do not be conversational. If the context contains the answer, synthesize it into a clear and direct response. If it does not, say "The provided notes do not contain this information."
Context:
{retrieved_chunks}

Question:
{user_query}

Answer:"""
