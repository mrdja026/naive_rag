"""
Retriever class for the RAG system.
Handles two-stage retrieval: initial ChromaDB search followed by cross-encoder reranking.
"""

import ollama
import chromadb
from sentence_transformers.cross_encoder import CrossEncoder
from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    RERANKER_MODEL_NAME,
    COLLECTION_NAME,
    TOP_N,
    CANDIDATES_TO_RETRIEVE
)


class Retriever:
    """Handles document retrieval and reranking for the RAG system."""
    
    def __init__(self):
        """Initialize ChromaDB client, collection, and reranker model."""
        print("Initializing Retriever...")
        
        # Initialize ChromaDB client and collection
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            print(f"Loaded existing collection with {self.collection.count()} documents.")
        except ValueError:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Please run ingest.py first to populate the database."
            )
        
        # Initialize reranker model
        print("Loading reranker model...")
        self.reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        print("Retriever initialization complete.")
    
    def _retrieve_candidates(self, query: str, top_n: int = CANDIDATES_TO_RETRIEVE) -> list:
        """
        Retrieve candidate documents from ChromaDB using cosine similarity.
        
        Args:
            query: The search query
            top_n: Number of candidates to retrieve
            
        Returns:
            List of tuples (document, similarity_score)
        """
        # Generate query embedding
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )
        
        # Process results
        retrieved_chunks = []
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for doc, dist in zip(documents, distances):
            similarity = 1 - dist  # Convert distance to similarity
            retrieved_chunks.append((doc, similarity))
        
        return retrieved_chunks
    
    def retrieve_and_rerank(self, query: str, top_n: int = TOP_N) -> list:
        """
        Perform two-stage retrieval: ChromaDB search followed by cross-encoder reranking.
        
        Args:
            query: The search query
            top_n: Number of final results to return
            
        Returns:
            List of tuples (document, reranker_score) sorted by relevance
        """
        # Stage 1: Retrieve candidates from ChromaDB
        candidate_docs = self._retrieve_candidates(query, top_n=CANDIDATES_TO_RETRIEVE)
        
        if not candidate_docs:
            return []
        
        # Stage 2: Rerank using cross-encoder
        candidate_chunks = [doc[0] for doc in candidate_docs]
        sentence_pairs = [[query, chunk] for chunk in candidate_chunks]
        
        # Get reranker scores
        reranker_scores = self.reranker_model.predict(sentence_pairs)
        
        # Combine chunks with scores and sort by relevance
        reranked_results = list(zip(candidate_chunks, reranker_scores))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_n]
