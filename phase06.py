# ## Phase 6 — Metadata Filtering

# **Task:** Add metadata to each document (e.g., `date`, `category`, `source`) and filter by it during retrieval.

# > **Hint:** FAISS doesn't support metadata natively — you'll need to store a parallel Python dict or list mapping each documen
# t index to its metadata. After retrieval, apply your filter *post-hoc* on the returned indices. If you switch to ChromaDB, 
# filtering becomes a first-class feature via the `where` parameter in `.query()`.

# ---

# ## Stretch: Evaluate Your System

# **Task:** Build a tiny eval loop. Create 5–10 (query, expected_answer) 
# pairs. For each one, run your RAG pipeline and check whether the right
#  document was retrieved.

# > **Hint:** This is Recall@K. For each query, does the ground-truth 
# document appear in your top-K results? Start with K=3. Print a score.
#  Then change your chunk size or overlap and re-run — does the score change?

# ---
