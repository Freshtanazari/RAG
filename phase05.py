# ## Phase 5 — Add a Confidence Threshold (Fallback)

# **Task:** Prevent hallucination when retrieval fails by checking
#  the similarity score before passing context to the LLM.

# > **Hint:** FAISS's `.search()` returns L2 *distances*, 
# not similarities — lower is better. You need to decide on a threshold
#  above which the retrieved document is "too far away" to be useful. 
# Try a few values and observe the behavior. What happens to your answer 
# if you pass irrelevant context to the LLM vs. returning a "I don't know" message?



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

# normalize the embeddings= true makes every vector unit length
document_embeddings = model.encode(document_chunks, normalize_embeddings=True)
print(document_embeddings.shape)

# store the embeddings in a FAISS index 
dimension = document_embeddings.shape[1]
# getting the inner product (IP) instead of the L2 == cosine similarity
index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)


# =========================================================
# STEP 3 — PREPARE THE QUESTION (both paths need it)
# =========================================================

# prepare the question 
question = "what is self-attention?"

#embed the question 
question_embedding = model.encode([question], normalize_embeddings=True)
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
#search and return similarities directly (higher = better)
similarities, indices = index.search(question_embedding, k=num_docs) 
# this returns two arrays: similarities that show similarity scores and 
# indices that shows the positions of the most similar docs in the original doc

semantic_scores = np.zeros(num_docs)
for sim, idx in zip(similarities[0], indices[0]):
    semantic_scores[idx] = sim

# adding the threshold to prevent hallucination
# instead of going with distances, i decided to go with the 
# similarity as i am already calculating it
best_similarity = similarities[0][0]
SIMILARITY_THRESHOLD = 0.24
print("Best similarity :", best_similarity)

if best_similarity < SIMILARITY_THRESHOLD: 
    print("I couldnt find the information in the documents")
    exit()

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

# our answer

#**Self‑attention** is an attention mechanism that operates *within a single sequence*.  
# In a self‑attention layer every position in the input sequence can look at every other
# position of that same sequence and combine their information.

# - For each token we create a *query*, a *key* and a *value* (all vectors of the same
#   dimensionality).
# - The compatibility of a query with a key is measured by the scaled dot‑product
#   \(QK^{T}/\sqrt{d_k}\).
# - A softmax over these scores gives the attention weights, which are then used to
#   take a weighted sum of the value vectors.
# - The result is a new representation for each token that incorporates information
#   from all tokens in the sequence.

# In the Transformer this operation is called **scaled dot‑product attention** and is
# performed in parallel for all tokens.  When the same mechanism is applied with
# different linear projections (heads) we obtain **multi‑head self‑attention**, which
# lets the model jointly attend to information from multiple representation sub‑spaces.


# bug faced: the similarity score i got using the formula couldnt capture the
# relevance of the document properly. 
# you can observe it in the two following question and their similarity score
# what is self-attention? similarity_score = 0.399  , result = exited the program  lower than threshold
# what is your name? similarity_score = 0.40135437 , result didnt exit the program slighly higher than threshold

# fixes: using cosine similarity instead of the distances, and lowering the threshold to 0.24

