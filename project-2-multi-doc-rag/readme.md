# Project 2: Multi-Document RAG with Metadata Filtering

A Retrieval-Augmented Generation (RAG) system that can search across multiple documents simultaneously, correctly attributing answers to their source, and using metadata filtering to guarantee balanced retrieval across sources.

## What it does
- Loads multiple PDFs, chunks them, and tags each chunk with its source filename (metadata)
- Stores all chunks in a single Chroma vector database
- Uses Groq's Llama 3.1 8B Instant (via API) for fast, high-quality answer generation
- Retrieves relevant chunks for a question and generates a grounded answer

## Key things I learned building this

**1. Metadata tagging enables source attribution**
Each chunk gets tagged with `metadata["source"] = filename` at chunking time, so retrieved answers can be traced back to which document they came from.

**2. Naive top-k retrieval doesn't guarantee balanced coverage across documents**
Asking a broad question like "What are the major topics covered in these documents?" returned chunks from *only one* of two loaded documents — even with `k=6`. This is because Chroma has no concept of "documents"; it purely ranks individual chunks by embedding similarity, and one document's phrasing style happened to match the question's wording more closely.

**Fix:** built separate retrievers per source using metadata filters, then manually combined a guaranteed number of chunks from each document before generating the answer:
```python
file_handling_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "filter": {"source": "doc1.pdf"}}
)
self_esteem_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "filter": {"source": "doc2.pdf"}}
)
```
This guarantees representation from every source for cross-document questions.

**3. Context window overflow can silently corrupt output**
In an earlier version of this pipeline (local flan-t5-base model), a combined prompt exceeding the token limit caused the question itself to get truncated, producing broken/repetitive output. Fixed by increasing max token length and reducing retrieved chunk count.

**4. API-based models (Groq/Llama 3.1) vastly outperform small local models (flan-t5-base) for RAG generation**
Same retrieval pipeline, dramatically better reasoning and answer quality when swapping the generation model.

## Stack
- LangChain (`langchain-classic` for `RetrievalQA`)
- ChromaDB (vector store)
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- Groq API (`llama-3.1-8b-instant`) for generation

## Next steps
- Compare chunking strategies (size/overlap) for retrieval quality
- Explore hybrid search (keyword + vector)
- Build a lightweight evaluation set to measure retrieval accuracy
