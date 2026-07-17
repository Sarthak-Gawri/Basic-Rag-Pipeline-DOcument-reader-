# Basic-Rag-Pipeline-DOcument-reader-

Spent the last day building my first RAG (Retrieval-Augmented Generation) pipeline from scratch — and broke it in every way possible along the way. Sharing what I learned, because the debugging taught me more than any tutorial could.
What I built: a "chat with your PDF" system — upload a document, ask questions, get answers grounded in the actual content (not the model just making things up).
The pipeline: PDF → chunk the text → convert chunks to embeddings → store in a vector database (Chroma) → retrieve the most relevant chunks for a question → feed them to an LLM to generate an answer.
Simple in theory. Here's what actually broke:
1️⃣ Retrieval failure — asked "what programming languages does this person know?" on my resume. The vector search pulled the wrong chunks entirely (matched "tech stack" instead of the actual "Languages" section). Learned that embedding similarity isn't perfect — it's a real, common failure mode in RAG systems.
2️⃣ Generation failure — even when retrieval worked, my small local LLM (flan-t5-base) just echoed back my contact info instead of answering. Learned that a RAG pipeline is only as strong as its weakest link — good retrieval means nothing if the model can't reason over it.
3️⃣ Context window overflow — tested on denser notes (my Theory of Computation material) and got a broken output of repeated asterisks. Turned out my combined prompt (retrieved chunks + question) was hitting ~501 tokens against a 512 token limit — the question itself was getting truncated before the model even saw it. Fixed by increasing max token length and reducing retrieved chunks.
The biggest lesson: RAG isn't magic — it's a pipeline with multiple points of failure (chunking, retrieval, generation, context limits), and debugging means isolating exactly which part broke instead of just staring at a bad final answer.
Next up: testing multi-document retrieval, comparing chunking strategies, and eventually fine-tuning.
