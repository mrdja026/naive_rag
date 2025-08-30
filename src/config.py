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
INSTRUCTION_PROMPT = """You are a specialized Q&A assistant for a personal research document. Your sole purpose is to answer questions by extracting and synthesizing information ONLY from the provided context.

Follow these rules strictly:
1.  **Grounding:** Base your answer exclusively on the text within the 'Context' section. Do not use any outside knowledge.
2.  **Extraction:** If the question asks for a link, list, or specific piece of information, extract it directly.
3.  **Honesty:** If the context does not contain the answer, you MUST state: "The answer is not available in the provided notes."
4.  **Conciseness:** Be direct and to the point. Avoid conversational filler.

Context:
{retrieved_chunks}

Question:
{user_query}

Answer:
"""
