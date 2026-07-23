# RAG Learning Journey

A hands-on journey learning Retrieval-Augmented Generation (RAG), vector databases, embeddings, and LLMs — building real, working pipelines from scratch, breaking them, and documenting what I learned from debugging each failure.

Each project folder contains its own notebook and README with a detailed writeup of what was built, what broke, and how it was fixed.

## Projects

### [Project 1: Chat with a PDF](./project-1-chat-with-pdf)
A basic RAG pipeline — upload a PDF, ask questions, get answers grounded in the document's content instead of the model just making things up.
- Built the full pipeline: chunking → embeddings → vector storage (Chroma) → retrieval → generation
- Diagnosed real failures: retrieval picking the wrong chunks, a weak local LLM (flan-t5-base) failing to reason over retrieved text, and a context-window overflow silently truncating the actual question from the prompt

### [Project 2: Multi-Document RAG with Metadata Filtering](./project-2-multi-doc-rag)
Extended the pipeline to search across multiple documents at once, with source attribution and guaranteed balanced retrieval.
- Tagged chunks with source metadata to trace answers back to their origin document
- Discovered that naive top-k similarity search doesn't guarantee coverage across multiple documents — fixed by building per-source retrievers and manually combining results
- Swapped the local flan-t5-base model for Groq's Llama 3.1 8B Instant API, seeing a major jump in answer quality

## Stack
- LangChain (`langchain-classic`, `langchain-community`, `langchain-groq`)
- ChromaDB (vector database)
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- Groq API (Llama 3.1 8B Instant) / local flan-t5-base (generation)

## What's next
- Comparing chunking strategies (size/overlap) for retrieval quality
- Hybrid search (keyword + vector retrieval)
- Building a small evaluation set to measure retrieval accuracy
- Fine-tuning basics
