#!/usr/bin/env python3
"""
Main entry point for the interactive RAG chat application.
Provides a command-line interface for asking questions and getting answers.

Usage:
    python main.py
"""

from retriever import Retriever
from generator import generate_answer, format_context


def main():
    """Main interactive chat function."""
    print("="*60)
    print("PRESENTATION RAG SYSTEM")
    print("="*60)
    print("Initializing system...")
    
    # Initialize the retriever
    try:
        retriever = Retriever()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please run 'python ingest.py' first to set up the database.")
        return
    
    print("System ready!")
    print("="*60)
    print("Ask me anything about the presentation! (type 'quit', 'exit', or 'q' to exit)")
    print("="*60)
    
    # Main interaction loop
    while True:
        try:
            user_query = input("\nYour question: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Skip empty queries
            if not user_query:
                continue
            
            # Retrieve relevant context
            print("Searching for relevant information...")
            relevant_chunks_data = retriever.retrieve_and_rerank(user_query)
            
            if not relevant_chunks_data:
                print("No relevant information found for your query.")
                continue
            
            # Format context
            context = format_context(relevant_chunks_data)
            
            # Generate and stream answer
            print("\nAnswer:")
            print("-" * 40)
            
            try:
                for chunk in generate_answer(context, user_query, stream=True):
                    print(chunk, end='', flush=True)
                print("\n" + "-" * 40)
                
            except Exception as e:
                print(f"Error generating answer: {e}")
                continue
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue


if __name__ == "__main__":
    main()
