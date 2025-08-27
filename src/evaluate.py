#!/usr/bin/env python3
"""
Standalone evaluation script for the RAG pipeline.
Evaluates the system's performance using the ragas framework.

Usage:
    python evaluate.py
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

from retriever import Retriever
from generator import generate_answer, format_context
from config import LANGUAGE_MODEL, EMBEDDING_MODEL


# Evaluation dataset
EVALUATION_DATASET = [
    {
        "question": "How long did it take to build the D&D agent?",
        "ground_truth_context": ["Time: 7 hours from zero to working product"],
        "ground_truth_answer": "It took 7 hours to build the D&D agent from zero to a working product."
    },
    {
        "question": "What analogy is used to explain a neural network?",
        "ground_truth_context": ['## Neural Network + KV Cache – "The Messy Library With a Map"'],
        "ground_truth_answer": "A neural network is compared to a 'messy library with a map'."
    },
    {
        "question": "What is the primary risk of the 'good enough' trap?",
        "ground_truth_context": ["\"Good enough\" trap: fast bootstrap can lead to tech debt if not revisited"],
        "ground_truth_answer": "The 'good enough' trap is that a fast bootstrap can lead to technical debt if the code is not revisited."
    },
    {
        "question": "According to the presentation, who is likely to take a developer's job?",
        "ground_truth_context": ["\"AI won't take your job. But the person who learns to pair program with it probably will.\""],
        "ground_truth_answer": "The presentation states that AI won't take your job, but a person who learns to pair program with it probably will."
    },
    {
        "question": "List three pitfalls of AI-assisted coding mentioned in the document.",
        "ground_truth_context": ["Maintainability: AI ≠ magic, code can rot without proper refactors", "Scalability: Not automatic; requires deliberate engineering", "Vulnerabilities: security, compliance, privacy"],
        "ground_truth_answer": "Three pitfalls mentioned are maintainability issues leading to code rot, scalability not being automatic, and security vulnerabilities."
    },
    {
        "question": "How did the narrative generation improve for the D&D agent?",
        "ground_truth_context": ["I made a move from Mistral dense to Mistral MoE → narrative generation was much better."],
        "ground_truth_answer": "The narrative generation improved by switching the model from a Mistral dense model to a Mistral MoE model."
    },
    {
        "question": "What is the presenter's call to action?",
        "ground_truth_context": ["Try one small feature or internal tool with AI this week", "Share your results & lessons with the team"],
        "ground_truth_answer": "The call to action is to try building a small feature or internal tool with AI this week and share the results and lessons with the team."
    },
    {
        "question": "How does the document describe the difference in output between GPT-5 and Claude?",
        "ground_truth_context": ["GPT-5 → more \"thinking aloud,\" creative detours", "Claude → more structured, but sometimes refused steps"],
        "ground_truth_answer": "GPT-5's output is described as more 'thinking aloud' with creative detours, while Claude's is more structured but sometimes refuses steps."
    },
    {
        "question": "What percentage of work does AI handle, and what is required for the rest?",
        "ground_truth_context": ["AI gets you ~80%, last 20% requires engineering judgment"],
        "ground_truth_answer": "AI gets you approximately 80% of the way, but the last 20% requires engineering judgment."
    },
    {
        "question": "What tasks was AI used for in the hotel booking project?",
        "ground_truth_context": ["AI used for: scaffolding, UI generation, copy tweaks"],
        "ground_truth_answer": "In the hotel booking project, AI was used for scaffolding, UI generation, and copy tweaks."
    },
    {
        "question": "Why is prompting described as a 'model-specific skill'?",
        "ground_truth_context": ["Prompting is model-specific skill — not portable 1:1"],
        "ground_truth_answer": "Prompting is described as a model-specific skill because it is not portable 1:1; what works in one model might fail in another."
    },
    {
        "question": "What features are explicitly mentioned as missing from the D&D agent?",
        "ground_truth_context": ["Missing: persistence layer (save/quit), multiplayer"],
        "ground_truth_answer": "The D&D agent is missing a persistence layer for saving and quitting, as well as a multiplayer feature."
    }
]


def main():
    """Main evaluation function."""
    print("Starting RAG system evaluation...")
    
    # Initialize components
    print("Initializing retriever...")
    try:
        retriever = Retriever()
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    print("Initializing evaluation models...")
    judge_llm = ChatOllama(model=LANGUAGE_MODEL)
    ragas_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Generate responses for evaluation
    print("Generating responses for evaluation...")
    responses = []
    contexts_list = []
    
    for i, item in enumerate(EVALUATION_DATASET):
        question = item['question']
        print(f"Processing question {i + 1}/{len(EVALUATION_DATASET)}: {question[:50]}...")
        
        # Retrieve relevant chunks
        relevant_chunks_data = retriever.retrieve_and_rerank(question)
        context = format_context(relevant_chunks_data)
        contexts_list.append([chunk[0] for chunk in relevant_chunks_data])
        
        # Generate answer
        response = generate_answer(context, question, stream=False)
        responses.append(response)
    
    # Create evaluation dataset
    print("Creating evaluation dataset...")
    eval_dataset = Dataset.from_dict({
        'question': [item['question'] for item in EVALUATION_DATASET],
        'answer': responses,
        'contexts': contexts_list,
        'ground_truth': [item['ground_truth_answer'] for item in EVALUATION_DATASET]
    })
    
    # Run evaluation
    print("Running RAGAS evaluation...")
    try:
        score = evaluate(
            eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=ragas_embeddings,
        )
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(score)
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
