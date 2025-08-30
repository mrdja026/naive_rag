"""
Configuration settings for the RAG system.
Contains all model names, file paths, and retrieval parameters.
"""

# --- MODEL CONFIGURATION ---
EMBEDDING_MODEL = 'hf.co/Fishkaras/embedic-large-Q8_0-GGUF:Q8_0'
LANGUAGE_MODEL = 'Mistral-7B-Instruct-v0.2-Q4_K_M:latest'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- FILE PATHS ---
CHROMA_DB_PATH = "./my_rag_db"
DATASET_PATH = "./obsidian_assets/main_discovery.txt"

# --- RETRIEVAL PARAMETERS ---
TOP_N = 3
CANDIDATES_TO_RETRIEVE = 10

# --- COLLECTION SETTINGS ---
COLLECTION_NAME = "my_presentation_docs"
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# --- PROMPT TEMPLATE ---
INSTRUCTION_PROMPT = """You are an expert assistant analyzing notes. Based *only* on the provided context, identify any unstated assumptions, missing elements, or unclear points that might require further attention or follow-up. Do not speculate beyond the context. If no gaps or missing points exist, say "No missing information found." 

Context:
{retrieved_chunks}

Task:
Review these notes carefully and list what is not explicitly covered or could be elaborated upon, focusing on potential gaps or points for clarification.

Answer:"""
