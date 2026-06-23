# ## Phase 0 — Environment Setup

# **Task:** Get your Python environment ready with the three core libraries the 
# tutorial's minimal example uses.

# > **Hint:** You need something for embeddings (sentence-level transformer models), 
# something for vector indexing (Facebook's similarity search library), 
# and an LLM API. The embedding model should run *locally* — look for the 
# `sentence-transformers` package on PyPI. The vector lib has "cpu" in its pip name.

# work: 

# run the following command: 
# pip install -U sentence-transformers from https://sbert.net/
# pip install faiss-cpu 
# Facebook's library for similarity search. 
# Once you embed your documents into vectors, 
# you need somewhere to store them and a way to find the most similar ones to a query.
# FAISS does that — it takes a query vector and quickly finds the closest document vectors in the index.
# The -cpu just means it runs on CPU rather than GPU.

# pip install groq
# lets you call the groq API from python

# import the API key
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("RAG_API_KEY")

# creating the sentence transformers model for embeddings


from sentence_transformers import SentenceTransformer

# an example:
# instantiate a model: 
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# embeddings = model.encode(["This is my first RAG project", "I am excited to learn more", "I try different ideas"])
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)

# cosine similarity is a measure of similarity between two non-zero vectors, that uses 
# cosine of the angle between them to detemine how similar they are based on their direction. 
# this is useful for comparing text embeddings, as it captures semantic similarity rather than just exact matches
# formula: 
# cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)

# how to use faiss: 
import faiss 

documents = ["The developer's name is Freshta Nazari", "My favorite sport is running", "I am testing my ideas"]
document_embeddings = model.encode(documents)
dimension = document_embeddings.shape[1] # this should match the dimension of your embeddings

# create an index with dimension
index = faiss.IndexFlatL2(dimension) 
# add to index 
index.add(document_embeddings)

# search the index with a query: get top k result
query= "wwhat is the developer's name?"
query_embedding = model.encode([query])

distances, indices = index.search(query_embedding, k=2)

# based on the indices we retrieve the actual documents from our original list
retrieved_docs = [documents[i] for i in indices[0]]

# construct the prompt for the llm
prompt = f"use only the following context to answer the question: context : {retrieved_docs}, question: {query}"

# how to use groq

from groq import Groq

client = Groq(api_key = api_key) 

response = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages = [{"role": "user", "content": prompt}]
)

answer = response.choices[0].message.content
print(answer)


