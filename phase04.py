# ## Phase 4 — Reranking
# Note: add the semantic chuncker

# **Task:** Take your top-20 retrieval results and 
# rerank them with a CrossEncoder, then return only the top 5.

# > **Hint:** A `CrossEncoder` is different from a `SentenceTransformer` —
#  it takes a *pair* `(query, document)` and outputs a single relevance score, 
# rather than independent embeddings. The `predict()` call takes a list of tuples.
#  Use `np.argsort(scores)[-5:]` to get the top 5 — but watch out: `argsort` is ascending by default.

# note: the cross encoder is less effiecent compared to the bi-encoder
# however it is more accurate that is why we use it only for the 20 top result 
# of ours from the bi-encoder


# =========================================================
# IMPORTS
# =========================================================

# --- pdf / env / llm ---
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq
import os

# --- chunking ---
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# --- bm25 (lexical) path ---
from rank_bm25 import BM25Okapi

# --- semantic (embedding) path ---
from sentence_transformers import SentenceTransformer
import faiss

# --- scoring / combining ---
import numpy as np
from sklearn.preprocessing import minmax_scale


# =========================================================
# STEP 1 — LOAD + CHUNK THE DOCUMENT (shared by both paths)
# =========================================================

reader = PdfReader("NIPS-2017-attention-is-all-you-need-Paper.pdf")

document_chunks = []

all_texts = ""
for page in reader.pages:
    all_texts += page.extract_text()

all_text = all_texts.split("Reference \n [1]")[0]

embedding_model = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-mpnet-base-v2" )

splitter = SemanticChunker(embedding_model)

documents = splitter.create_documents([all_text])
document_chunks = [doc.page_content for doc in documents]


# =========================================================
# STEP 2A — BM25 (LEXICAL) PATH
# =========================================================

tokenized_docs = [doc.split() for doc in document_chunks]
bm25 = BM25Okapi(tokenized_docs)


# =========================================================
# STEP 2B — SEMANTIC (EMBEDDING) PATH
# =========================================================

# embed the documents
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
document_embeddings = model.encode(document_chunks)
print(document_embeddings.shape)

# store the embeddings in a FAISS index 
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)


# =========================================================
# STEP 3 — PREPARE THE QUESTION (both paths need it)
# =========================================================

# prepare the question 
question = "Why Attention is important?"

#embed the question 
question_embedding = model.encode([question])
#tokenize for bm25 search as well 
tokenized_query = question.split()


# =========================================================
# STEP 4A — SCORE WITH BM25
# =========================================================

bm25_scores = bm25.get_scores(tokenized_query)


# =========================================================
# STEP 4B — SCORE WITH SEMANTIC SEARCH (FAISS)
# =========================================================

#get the number of docs 
num_docs = len(document_chunks)
#search the index for all the docs 
distances, indices = index.search(question_embedding, k=num_docs)
# this returns two arrays: distances that show similarity scores and 
# indices that shows the positions of the most similar docs in the original doc

semantic_scores = np.zeros(num_docs)

#change the distance to similarity so that the higher
#score the better the result
for dist, idx in zip(distances[0], indices[0]):
    semantic_scores[idx] = 1 / (1+dist)


# =========================================================
# STEP 5 — COMBINE BOTH PATHS INTO HYBRID SCORES
# =========================================================

nor_bm25_scores = minmax_scale(bm25_scores)
nor_semantic_scores = minmax_scale(semantic_scores)

#hybrid_scores of the two 
# use the weighted linear combination using the formula: 
#hybrid_score(d)=α⋅s^sem(d)+(1−α)⋅s^bm25(d)

alpha = 0.5 # each method (semantic and bm25) contributes equally
hybrid_scores = alpha * nor_semantic_scores + (1-alpha) * nor_bm25_scores

# get the top 20 of the hybrid scores
top_k = 20
hybrid_indices = np.argsort(hybrid_scores)[::-1][:top_k]

#retrieve the actual documents based on the indices returned by FAISS
similar_docs = [document_chunks[i] for i in hybrid_indices]

# =========================================================
# STEP 6 — CrossEncoder
# =========================================================
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# pair the question and the top 20 docs
query_doc = [(question, doc) for doc in similar_docs]

# get the score of cross encoding them 
cross_encoder_scores = cross_encoder.predict(query_doc)

#sort the scores and get their indices 
cross_encoder_top5 = np.argsort(cross_encoder_scores)[-5:]

#get the actual doc of the top 5 
cross_encoder_top5_doc = [similar_docs[i] for i in cross_encoder_top5 ]

# =========================================================
# STEP 7 — PROMPT + LLM CALL
# =========================================================

# construct the prompt for the llm 
prompt = f"Use only the following context to answer the question: context: {cross_encoder_top5_doc}, question: {question}"

# initialize the groq client with the API key 
load_dotenv()
api_key = os.getenv("RAG_API_KEY")

client = Groq(api_key = api_key)

response = client.chat.completions.create(
    model= "openai/gpt-oss-20b", 
    messages = [{"role":"user", "content": prompt}]
)

answer = response.choices[0].message.content
print(answer)


# our answer with cross encoder: 
#**Attention is important because it gives a neural model the ability to capture long‑range dependencies, to do so efficiently and in a way that is fully parallelisable, and to focus selectively on the most relevant parts of the input when generating each output token.**

# 1. **Global dependency modelling**  
#    - Classical recurrent or convolutional architectures compute representations one step at a time, which makes it hard for the network to “reach” information that is far away in the sequence.  
#    - Attention mechanisms compute a weighted sum over *all* positions (Equation (1) in the paper).  This means that, when predicting a particular output symbol, the model can directly consult any part of the source sentence, regardless of its distance, and assign it the appropriate weight.  The paper notes that “attention mechanisms … allow modelling of dependencies without regard to their distance” and that this property is crucial for tasks such as translation where long‑range word alignments are common.

# 2. **Parallelism and training speed**  
#    - Because attention does not rely on sequential recurrence, all positions can be processed simultaneously.  The Transformer can therefore be trained much faster: the paper reports training a state‑of‑the‑art English‑to‑German model in just 12 h on 8 GPUs, far quicker than comparable recurrent or convolutional baselines.  
#    - The paper also contrasts attention with convolutional approaches, noting that convolution requires many layers to span a long distance (“requiring a stack of O(n/k) convolutional layers”), whereas a single attention layer achieves a constant‑time connection between any two positions.

# 3. **Improved performance**  
#    - Empirically, the Transformer, which is built solely on multi‑head attention and position‑wise feed‑forward layers, outperforms all previously published single‑model translation systems (28.4 BLEU for English‑to‑German and 41.0 BLEU for English‑to‑French).  
#    - The paper also shows that different attention heads specialise in different linguistic roles (e.g., syntactic or semantic patterns), providing richer representations than a single recurrent path.

# 4. **Simplicity and modularity**  
#    - Attention layers are simple to implement and can be stacked in a modular fashion.  The Transformer uses only a few core components: scaled dot‑product attention, multi‑head attention, and a point‑wise feed‑forward network.  This simplicity allows easy experimentation with architectural changes and makes the model easier to optimise.

# In short, attention is important because it gives neural models the capacity to look globally across a sequence, does so in a way that scales linearly with sequence length and can be computed in parallel, and yields higher‑quality results on hard sequence‑to‑sequence tasks.