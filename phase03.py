# ## Phase 3 — Hybrid Search

# **Task:** Augment your retrieval with BM25 alongside the
#  semantic search and combine the results.

# > **Hint:** Install `rank_bm25`. BM25 works on tokenized text (split by whitespace).
#  It returns a *score* per document, not indices — you'll need `np.argsort` to rank them. 
# The tricky part is merging two ranked lists with different score scales.
#  Start with equal weighting (`alpha=0.5`) and normalize each list's scores to [0, 1] 
# before combining.

# **Reflection question:** Try the query "automobile insurance"
#  against a document set that contains the word "car" but never 
# "automobile". Which method retrieves it — semantic or BM25? Why?

# ---
# what is Best matching 25?
# an improved version of TF-IDF(term frequency-inverse Document Frequency)
# normalizes docs, matching for exact term or near-term matches

# install BM25
# pip install rank-bm25

from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import os 
from dotenv import load_dotenv 
import requests
from pypdf import PdfReader
# import the BM25Okapi
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.preprocessing import minmax_scale


# for semantic chunking isntall the following
#pip install langchain-experimental 
# pip install langchain-huggingface
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# The following pdf is a research paper titled: Attention is all you need
reader = PdfReader("NIPS-2017-attention-is-all-you-need-Paper.pdf")
   
documents_chunks = []

all_text =""
for page in reader.pages:
    all_text += page.extract_text()
all_text = all_text.split("Reference \n [1]")[0]

# semantic chunking
# create the embedding model for the chunker
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
# create the semantic chunker
splitter = SemanticChunker(embeddings)
# generate the chunks
documents = splitter.create_documents([all_text])
documents_chunks = [doc.page_content for doc in documents]

# tokenize each chunk
tokenized_docs = [doc.split() for doc in documents_chunks]
# create BM25 index to store the token statistics
bm25 = BM25Okapi(tokenized_docs)


# embed the documents
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
document_embeddings = model.encode(documents_chunks)
print(document_embeddings.shape)

# store the embeddings in a FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# prepare the question 
question = "Why Attention is important?"

#embed the question
question_embedding = model.encode([question])
#tokenize for bm25 search 
tokenized_query = question.split()
bm25_scores = bm25.get_scores(tokenized_query)

# get the number of docs
num_docs = len(documents_chunks)
# search the index for all the docs  
distances, indices = index.search(question_embedding, k=num_docs)
# this returns two arrays: distances that shows similarity scores and 
# indices that shows the positions of the most similar docs in the original doc
semantic_scores = np.zeros(num_docs)

# change the distance to similarity so that the higher score the better result
for dist, idx in zip(distances[0], indices[0]):
    semantic_scores[idx] = 1 / (1+dist)

nor_bm25_scores = minmax_scale(bm25_scores)
nor_semantic_scores = minmax_scale(semantic_scores)

# hybrid_scores of the two
# use the weighted linear combination using the formula: 
#hybrid_score(d)=α⋅s^sem​(d)+(1−α)⋅s^bm25​(d)
alpha = 0.5 # each method (semantic and bm25) contribute equally
hybrid_scores = alpha * nor_semantic_scores + (1-alpha) * nor_bm25_scores

# get the top k of the hybrid scores
top_k = 3
hybrid_indices = np.argsort(hybrid_scores)[::-1][:top_k]

# retrieve the actual documents based on the indices returned by FAISS 
similar_docs = [documents_chunks[i] for i in hybrid_indices]

# construct the promopt for the llm 
prompt = f"Use only the following context to answer the question: context: {similar_docs}, question: {question}"

# initialize the groq client with the API key 
load_dotenv()
api_key = os.getenv("RAG_API_KEY")

client = Groq(api_key = api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant", 
    messages=[{"role":"user", "content": prompt}]
)

answer = response.choices[0].message.content
print(answer)


# the answer returned:
# According to the paper "Attention Is All You Need" by Vaswani et al., attention is important for several reasons:

# 1. **Efficient Computation**: Attention mechanisms, such as self-attention, enable the model to connect all positions in the input and output sequences with a constant number of sequentially executed operations, which is much faster than the O(n) sequential operations required by recurrent layers.
# 2. **Flexible Dependency Modeling**: Attention allows the model to jointly attend to information from different representation subspaces at different positions, which is beneficial for modeling long-range dependencies.
# 3. **Reduced Computational Complexity**: Self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is often the case in machine translation tasks.
# 4. **Improved Path Length**: Self-attention layers connect all positions with a constant number of sequentially executed operations, which makes it easier to learn long-range dependencies compared to recurrent layers, which require O(n) sequential operations.
# 5. **Enhanced Parallelization**: Attention mechanisms enable more parallelization, making it easier to train large-scale models.
# 6. **Better Modeling of Global Dependencies**: Attention mechanisms can model global dependencies without regard to their distance in the input or output sequences, which is beneficial for many sequence modeling tasks.

# In summary, attention is important because it enables efficient computation, flexible dependency modeling, reduced computational complexity, improved path length, enhanced parallelization, and better modeling of global dependencies.

