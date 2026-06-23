# # lets build an RAG based on this guide:
# links: 
# https://github.com/langchain-ai/rag-from-scratch/tree/main
# https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i
# # Good, I've read the full article. Here's your guided challenge — structured by phase, with hints that push you to think rather than copy.

# ---

# # RAG Build Guide: Hints Only 🧠

# Given your background (data science cert + CS + web dev), I'll pitch this at the right level — you should be able to reason your way through each step, but the hints will prevent you from going in circles.

# ---

# ## Phase 0 — Environment Setup

# **Task:** Get your Python environment ready with the three core libraries the tutorial's minimal example uses.

# > **Hint:** You need something for embeddings (sentence-level transformer models), something for vector indexing (Facebook's similarity search library), and an LLM API. The embedding model should run *locally* — look for the `sentence-transformers` package on PyPI. The vector lib has "cpu" in its pip name.

# ---

# ## Phase 1 — Build the Minimal RAG (50 lines)

# **Task:** Create a small document list (5–10 sentences on any topic you know well), embed them, store in an index, retrieve the top-2 most similar to a query, and pass those to an LLM to generate an answer.

# ### Sub-challenges:

# **1a. Embedding**
# > **Hint:** `SentenceTransformer` takes a model name string. The model used in the tutorial is lightweight (~22MB). Call `.encode()` on a list of strings — what does the output shape look like? What does each row represent?

# **1b. Indexing**
# > **Hint:** FAISS needs to know the *dimensionality* of your vectors before you add anything. That dimension comes from the embedding model. `IndexFlatL2` uses Euclidean distance — think about what that means geometrically for similarity.

# **1c. Retrieval**
# > **Hint:** You encode the *query* the same way you encoded the documents. Then `.search(query_embedding, k)` returns two arrays — what do they represent? How do you map indices back to your original document strings?

# **1d. Generation**
# > **Hint:** Build the prompt as a Python f-string. The key prompt engineering insight here: you want the LLM to answer *only from the context*, not from its parametric memory. How would you phrase that instruction?

# **Stop point:** Test with 3 different queries. Does retrieval feel right? If not, why might it be wrong before you even touch the LLM?

# ---

# ## Phase 2 — Chunking Experiments

# **Task:** Take a longer document (grab a Wikipedia article or any multi-paragraph text) and implement the two chunking strategies from the tutorial, then compare results.

# **2a. Fixed-size chunking**
# > **Hint:** This is a sliding window over a word list. Two parameters: window size and overlap. If `chunk_size=512` and `overlap=50`, what's the step size in your loop? Make sure the last chunk doesn't get silently dropped.

# **2b. Semantic chunking (LangChain)**
# > **Hint:** `RecursiveCharacterTextSplitter` tries separators *in order* — `\n\n` first, then `\n`, then `.`. Think about why that order makes semantic sense. The `chunk_overlap` parameter here works on characters, not words.

# **Reflection question (no code):** Given two chunks of the same document — one that cuts mid-sentence and one that ends at a paragraph break — which one will embed more accurately, and why? What does that imply for retrieval quality?

# ---

# ## Phase 3 — Hybrid Search

# **Task:** Augment your retrieval with BM25 alongside the semantic search and combine the results.

# > **Hint:** Install `rank_bm25`. BM25 works on tokenized text (split by whitespace). It returns a *score* per document, not indices — you'll need `np.argsort` to rank them. The tricky part is merging two ranked lists with different score scales. Start with equal weighting (`alpha=0.5`) and normalize each list's scores to [0, 1] before combining.

# **Reflection question:** Try the query "automobile insurance" against a document set that contains the word "car" but never "automobile". Which method retrieves it — semantic or BM25? Why?

# ---

# ## Phase 4 — Reranking

# **Task:** Take your top-20 retrieval results and rerank them with a CrossEncoder, then return only the top 5.

# > **Hint:** A `CrossEncoder` is different from a `SentenceTransformer` — it takes a *pair* `(query, document)` and outputs a single relevance score, rather than independent embeddings. The `predict()` call takes a list of tuples. Use `np.argsort(scores)[-5:]` to get the top 5 — but watch out: `argsort` is ascending by default.

# ---

# ## Phase 5 — Add a Confidence Threshold (Fallback)

# **Task:** Prevent hallucination when retrieval fails by checking the similarity score before passing context to the LLM.

# > **Hint:** FAISS's `.search()` returns L2 *distances*, not similarities — lower is better. You need to decide on a threshold above which the retrieved document is "too far away" to be useful. Try a few values and observe the behavior. What happens to your answer if you pass irrelevant context to the LLM vs. returning a "I don't know" message?

# ---

# ## Phase 6 — Metadata Filtering

# **Task:** Add metadata to each document (e.g., `date`, `category`, `source`) and filter by it during retrieval.

# > **Hint:** FAISS doesn't support metadata natively — you'll need to store a parallel Python dict or list mapping each document index to its metadata. After retrieval, apply your filter *post-hoc* on the returned indices. If you switch to ChromaDB, filtering becomes a first-class feature via the `where` parameter in `.query()`.

# ---

# ## Stretch: Evaluate Your System

# **Task:** Build a tiny eval loop. Create 5–10 (query, expected_answer) pairs. For each one, run your RAG pipeline and check whether the right document was retrieved.

# > **Hint:** This is Recall@K. For each query, does the ground-truth document appear in your top-K results? Start with K=3. Print a score. Then change your chunk size or overlap and re-run — does the score change?

# ---

# ## Mental Model Checkpoint

# Before moving to agentic RAG or production concerns, make sure you can answer these without code:

# - Why does chunking with overlap help retrieval?
# - Why is a CrossEncoder more accurate than cosine similarity between independent embeddings — and why is it also slower?
# - Why would a 7B local model + RAG sometimes beat GPT-4 on a narrow domain task?
# - What breaks first when you scale from 100 documents to 1 million?

# If those feel solid, you're ready for the agentic / production section. Want me to prep a challenge guide for that too?