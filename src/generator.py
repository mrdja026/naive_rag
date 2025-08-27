"""
Answer generation module for the RAG system.
Contains the generate_answer function that formats prompts and calls the LLM.
"""

import ollama
from config import LANGUAGE_MODEL, INSTRUCTION_PROMPT


def generate_answer(context: str, query: str, stream: bool = True):
    """
    Generate an answer using the language model based on retrieved context and user query.
    
    Args:
        context: The retrieved and reranked context chunks as a single string
        query: The user's question
        stream: Whether to stream the response (default: True)
        
    Returns:
        If stream=True: Returns a generator that yields response chunks
        If stream=False: Returns the complete response as a string
    """
    # Format the instruction prompt with context and query
    formatted_prompt = INSTRUCTION_PROMPT.format(
        retrieved_chunks=context,
        user_query=query
    )
    
    # Call the language model
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'user', 'content': formatted_prompt}],
        stream=stream,
    )
    
    if stream:
        # Return generator for streaming responses
        for chunk in response:
            yield chunk['message']['content']
    else:
        # Return complete response for evaluation
        return response['message']['content']


def format_context(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks into a single context string.
    
    Args:
        retrieved_chunks: List of tuples (document, score)
        
    Returns:
        Formatted context string
    """
    return "\n".join([chunk[0] for chunk in retrieved_chunks])
