# ## Phase 6 — Metadata Filtering

# **Task:** Add metadata to each document (e.g., `date`, `category`, `source`) and filter by it during retrieval.

# > **Hint:** FAISS doesn't support metadata natively 
# — you'll need to store a parallel Python dict or list
#  mapping each documen
# t index to its metadata. After retrieval, apply your
#  filter *post-hoc* on the returned indices. If you switch to ChromaDB, 
# filtering becomes a first-class feature via the `where` parameter in `.query()`.

# ---

# ## Stretch: Evaluate Your System

# **Task:** Build a tiny eval loop. Create 5–10 (query, expected_answer) 
# pairs. For each one, run your RAG pipeline and check whether the right
#  document was retrieved.

# > **Hint:** This is Recall@K. For each query, does the ground-truth 
# document appear in your top-K results? Start with K=3. Print a score.
#  Then change your chunk size or overlap and re-run — does the score change?


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

# importing path
from pathlib import Path

from datetime import date
# cross encoder
from sentence_transformers import CrossEncoder




# # load the model and return a splitter

def load_model_chunker(model_name):
   embedding_model = HuggingFaceEmbeddings(model_name = model_name)
   splitter = SemanticChunker(embedding_model)
   cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   return splitter, cross_encoder


# extract and chunk the data

def load_and_chunk(file_path, splitter):
    metadata = []
    document_chunks = []

    for file in file_path.glob("*.pdf"):
        reader = PdfReader(file)
        all_texts = ""
        for page in reader.pages:
            all_texts += page.extract_text()

        documents = splitter.create_documents([all_texts])
        for doc in documents:
            doc.metadata["source"] = file.name
            doc.metadata["date"] = date.today()
            metadata.append(doc.metadata)

        document_chunks.extend([doc.page_content for doc in documents])

    return (metadata, document_chunks)

# =========================================================
# STEP 2A — BM25 (LEXICAL) PATH
# =========================================================

def get_bm25_score(document_chunks):
   tokenized_docs = [doc.split() for doc in document_chunks]
   bm25 = BM25Okapi(tokenized_docs)
   return bm25


# =========================================================
# STEP 2B — SEMANTIC (EMBEDDING) PATH
# =========================================================

def get_doc_semantic_index(document_chunks):
   # embed the documents
   model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
   # normalize the embeddings= true makes every vector unit length
   document_embeddings = model.encode(document_chunks, normalize_embeddings=True)
   # store the embeddings in a FAISS index 
   dimension = document_embeddings.shape[1]
   # getting the inner product (IP) instead of the L2 == cosine similarity
   index = faiss.IndexFlatIP(dimension)
   index.add(document_embeddings)
   return index, model

# =========================================================
# STEP 3 — PREPARE THE QUESTION (both paths need it)
# =========================================================
def prepare_query(question, model):
   #embed the question 
   question_embedding = model.encode([question], normalize_embeddings=True)
   #tokenize for bm25 search as well 
   tokenized_query = question.split()
   return tokenized_query, question_embedding


def get_both_scores(bm25, tokenized_query, document_chunks, question_embedding, index):
    # =========================================================
    # STEP 4A — SCORE WITH BM25
    # =========================================================

    bm25_scores = bm25.get_scores(tokenized_query)

    # =========================================================
    # STEP 4B — SCORE WITH SEMANTIC SEARCH (FAISS)
    # =========================================================

    # get the number of docs
    num_docs = len(document_chunks)

    # search and return similarities directly (higher = better)
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
        return (None, None)

    return bm25_scores, semantic_scores



# =========================================================
# STEP 5 — COMBINE BOTH PATHS INTO HYBRID SCORES
# =========================================================

def get_hybrid_scores(bm25_scores, semantic_scores, document_chunks, metadata, valid_sources):
    if bm25_scores is None or semantic_scores is None:
        return (None, None)
    nor_bm25_scores = minmax_scale(bm25_scores)
    nor_semantic_scores = minmax_scale(semantic_scores)

    # hybrid_scores of the two
    # use the weighted linear combination using the formula:
    # hybrid_score(d)=α⋅s^sem(d)+(1−α)⋅s^bm25(d)

    alpha = 0.5  # each method (semantic and bm25) contributes equally
    hybrid_scores = alpha * nor_semantic_scores + (1 - alpha) * nor_bm25_scores

    # get the top 20 of the hybrid scores
    top_k = 20
    hybrid_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    similar_docs = []
    similar_indices =[]
    # filtering only the requested documents
    if len (valid_sources) >=  1:
        for index in hybrid_indices:
            if metadata[index]["source"] in valid_sources:
                similar_docs.append(document_chunks[index])
                similar_indices.append(index)

    if len(similar_docs) < 1:
        print("No such document has been found")
        return (None, None)

    return similar_docs, similar_indices

# =========================================================
# STEP 6 — CrossEncoder
# =========================================================

def cross_encode(question, similar_docs, similar_indices, k, cross_encoder):
    if similar_docs is None:
        return (None, None)
    
    # pair the question and the top docs
    query_doc = [(question, doc) for doc in similar_docs]

    # get the score of cross encoding them
    cross_encoder_scores = cross_encoder.predict(query_doc)

    # sort the scores and get their indices
    cross_encoder_top5 = np.argsort(cross_encoder_scores)[-k:]

    # get the actual doc of the top 5
    cross_encoder_top5_doc = [similar_docs[i] for i in cross_encoder_top5]
    top5_indices = [similar_indices[i] for i in cross_encoder_top5]
    return cross_encoder_top5_doc, top5_indices


# =========================================================
# STEP 7 — PROMPT + LLM CALL
# =========================================================

def prompt_llm(top_5, question):
    if top_5 is None:
        return "No Similar documents have been found"
    # construct the prompt for the llm
    prompt = f"Use only the following context to answer the question: context: {top_5}, question: {question}"

    # initialize the groq client with the API key
    load_dotenv()
    api_key = os.getenv("RAG_API_KEY")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return answer

# def run_pipeline(question, sources):
def run_pipeline(question, valid_sources, document_chunks, metadata, bm25, index, model, k, crossEncoder):
    tokenized_query, question_embedding = prepare_query(question, model)
    bm25_scores, semantic_scores = get_both_scores(bm25, tokenized_query, document_chunks, question_embedding, index)
    similar_docs, similar_indices = get_hybrid_scores(bm25_scores, semantic_scores, document_chunks, metadata, valid_sources)
    top_5, top5_indices = cross_encode(question, similar_docs, similar_indices, k, crossEncoder)
    return prompt_llm(top_5, question)


def retrieve_sources(question, valid_sources, document_chunks, metadata, bm25, index, model, k, crossEncoder):
    tokenized_query, question_embedding = prepare_query(question, model)
    bm25_scores, semantic_scores = get_both_scores(bm25, tokenized_query, document_chunks, question_embedding,index)
    similar_docs, similar_indice = get_hybrid_scores(bm25_scores, semantic_scores, document_chunks, metadata, valid_sources)
    top_5, top5_indices = cross_encode(question, similar_docs, similar_indice, k, crossEncoder)
    if top5_indices is None:
        return []
    return [metadata[i]["source"] for i in top5_indices]

