# Naive RAG System

A modular Retrieval-Augmented Generation (RAG) system built for learning purposes. This project demonstrates how to build a professional, maintainable RAG pipeline with clear separation of concerns.

## 🎯 Learning Objectives

This project serves as a hands-on learning experience for:

- Building modular ML/AI applications
- Understanding RAG architecture and components
- Implementing two-stage retrieval (vector search + reranking)
- Evaluating RAG system performance
- Creating maintainable Python codebases

## 🏗️ Project Structure

```
src/
├── config.py          # Configuration variables and constants
├── ingest.py          # Standalone data ingestion script
├── retriever.py       # Retriever class with two-stage retrieval
├── generator.py       # Answer generation functions
├── evaluate.py        # Standalone evaluation script
├── main.py           # Interactive chat application entry point
└── requirements.txt   # Python package dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Required models downloaded in Ollama

### Installation

1. **Install dependencies:**

   ```bash
   cd src
   pip install -r requirements.txt
   ```

2. **Download required Ollama models:**
   ```bash
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ollama pull Mistral-7B-Instruct-v0.2-Q4_K_M:latest
   ```

## 📋 How to Use Each Component

### 1. `ingest.py` - Data Ingestion

**Purpose:** Populates the ChromaDB database with document embeddings (run once).

**Usage:**

```bash
python ingest.py
```

**What it does:**

- Reads the dataset from `../assets/dataset.txt`
- Generates embeddings using the configured embedding model
- Stores documents and embeddings in ChromaDB
- Shows progress indicators during processing
- Skips ingestion if data already exists

**When to run:**

- First time setup
- When you want to re-ingest data (delete the database folder first)
- After updating the dataset

---

### 2. `main.py` - Interactive Chat

**Purpose:** Main entry point for asking questions and getting answers.

**Usage:**

```bash
python main.py
```

**What it does:**

- Initializes the retriever system
- Provides an interactive command-line interface
- Processes user questions through the full RAG pipeline
- Streams answers in real-time
- Handles exit commands (`quit`, `exit`, `q`)

**Example interaction:**

```
Your question: How long did it take to build the D&D agent?
Answer:
----------------------------------------
It took 7 hours to build the D&D agent from zero to a working product.
----------------------------------------
```

---

### 3. `evaluate.py` - System Evaluation

**Purpose:** Evaluates RAG system performance using the RAGAS framework.

**Usage:**

```bash
python evaluate.py
```

**What it does:**

- Runs predefined evaluation questions through the system
- Measures performance using RAGAS metrics:
  - **Faithfulness:** How accurate answers are to the retrieved context
  - **Answer Relevancy:** How well answers address the questions
  - **Context Precision:** Quality of retrieved context
  - **Context Recall:** Completeness of retrieved context
- Generates comprehensive evaluation scores

**Sample output:**

```
EVALUATION RESULTS
============================================================
{'faithfulness': 0.85, 'answer_relevancy': 0.78, 'context_precision': 0.82, 'context_recall': 0.75}
============================================================
```

---

### 4. Individual Modules (For Development)

#### `config.py`

Contains all configuration settings. Modify this file to:

- Change model names
- Adjust retrieval parameters
- Update file paths

#### `retriever.py`

Contains the `Retriever` class. Can be imported and used standalone:

```python
from retriever import Retriever

retriever = Retriever()
results = retriever.retrieve_and_rerank("your question")
```

#### `generator.py`

Contains answer generation functions. Can be used independently:

```python
from generator import generate_answer, format_context

context = "your context here"
query = "your question"

# Streaming response
for chunk in generate_answer(context, query, stream=True):
    print(chunk, end='')

# Complete response
response = generate_answer(context, query, stream=False)
```

## 🔄 Typical Workflow

1. **Setup (once):**

   ```bash
   pip install -r requirements.txt
   python ingest.py
   ```

2. **Interactive testing:**

   ```bash
   python main.py
   ```

3. **Performance evaluation:**

   ```bash
   python evaluate.py
   ```

4. **Development cycle:**
   - Modify configuration in `config.py`
   - Test changes with `python main.py`
   - Evaluate performance with `python evaluate.py`
   - Re-ingest data if needed: `python ingest.py`

## 🛠️ Customization

### Adding New Evaluation Questions

Edit the `EVALUATION_DATASET` in `evaluate.py`:

```python
{
    "question": "Your new question?",
    "ground_truth_context": ["Expected context"],
    "ground_truth_answer": "Expected answer"
}
```

### Changing Models

Update `config.py`:

```python
EMBEDDING_MODEL = 'your-embedding-model'
LANGUAGE_MODEL = 'your-language-model'
RERANKER_MODEL_NAME = 'your-reranker-model'
```

### Adjusting Retrieval Parameters

Modify in `config.py`:

```python
TOP_N = 5  # Number of final results
CANDIDATES_TO_RETRIEVE = 20  # Initial candidates
```

## 🐛 Troubleshooting

**"Collection not found" error:**

- Run `python ingest.py` first

**Model not found:**

- Ensure Ollama models are downloaded
- Check model names in `config.py`

**Import errors:**

- Ensure you're in the `src/` directory
- Check that all dependencies are installed

## 📚 Learning Resources

This project demonstrates:

- **Modular design patterns** in ML applications
- **Two-stage retrieval** (dense retrieval + reranking)
- **RAG evaluation** using established metrics
- **Error handling** and user experience design
- **Configuration management** for ML systems

## 🎓 Next Steps

After understanding this system, consider:

- Adding more sophisticated chunking strategies
- Implementing query expansion techniques
- Adding conversation memory
- Creating a web interface
- Exploring different embedding models
- Adding vector database alternatives
