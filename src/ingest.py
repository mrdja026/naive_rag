#!/usr/bin/env python3
"""
Standalone data ingestion script for the RAG system.
Run this script once to populate the ChromaDB database with embeddings.

Usage:
    python ingest.py
"""

import ollama
import chromadb
from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    DATASET_PATH,
    COLLECTION_NAME,
    COLLECTION_METADATA
)


def main():
    """Main ingestion function."""
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA
    )
    
    # Check if data already exists
    if collection.count() > 0:
        print(f"Database already contains {collection.count()} chunks. Skipping ingestion.")
        print("To re-ingest data, delete the database directory first.")
        return
    
    print("No existing data found in DB. Starting ingestion...")
    
    # Read and prepare documents
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            dataset_lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        print("Please ensure the dataset file exists before running ingestion.")
        return
    
    documents = [chunk.strip() for chunk in dataset_lines if chunk.strip()]
    
    if not documents:
        print("Error: No valid documents found in dataset file.")
        return
    
    print(f"Found {len(documents)} documents to process...")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings_list = []
    ids = [str(i) for i in range(len(documents))]
    
    for i, doc in enumerate(documents):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing document {i + 1}/{len(documents)}")
        
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=doc)['embeddings'][0]
            embeddings_list.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for document {i}: {e}")
            return
    
    # Add to collection
    print("Adding documents to ChromaDB...")
    try:
        collection.add(
            embeddings=embeddings_list,
            documents=documents,
            ids=ids
        )
        print(f"Successfully ingested {len(documents)} chunks into ChromaDB.")
        print(f"Database saved to: {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"Error adding documents to collection: {e}")
        return


if __name__ == "__main__":
    main()
