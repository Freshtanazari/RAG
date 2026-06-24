# ## Phase 2 — Chunking Experiments

# **Task:** Take a longer document (grab a Wikipedia article or any multi-paragraph text) and
#  implement the two chunking strategies from the tutorial, then compare results.

# **2a. Fixed-size chunking**
# > **Hint:** This is a sliding window over a word list. Two parameters: window size and overlap. 
# If `chunk_size=512` and `overlap=50`, what's the step size in your loop?
#  Make sure the last chunk doesn't get silently dropped.

# **2b. Semantic chunking (LangChain)**
# > **Hint:** `RecursiveCharacterTextSplitter` tries separators *in order* — `\n\n` first, then `\n`, then `.`. Think about why that order makes semantic sense. 
# The `chunk_overlap` parameter here works on characters, not words.

# **Reflection question (no code):** Given two chunks of the same document — one that cuts mid-sentence 
# and one that ends at a paragraph break — which one will embed more accurately, and why? What does that
#  imply for retrieval quality?

# note: pip install pypdf


# ---
# A list of strings detailing a standard corporate procedure for handling customer complaints
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import os 
from dotenv import load_dotenv 
import requests
from pypdf import PdfReader


# The following pdf is a research paper titled: Attention is all you need
reader = PdfReader("NIPS-2017-attention-is-all-you-need-Paper.pdf")
pages =[]
text = ""

for page in reader.pages:
    text = page.extract_text()
    if "Reference \n [1]" in text:
        text = text.split("Reference \n [1]")[0]
        pages.append(text)
        break 
    pages.append(text)
   
documents_chunks = []


def fixed_size_chunks(chunk_size, overlap, documents):
    # documents should be an array of strings
    chunk_size = chunk_size - overlap
    for page in documents:
        current_size = 0
        while current_size < len(page):
            chunk = page[current_size - overlap if current_size > overlap else current_size:current_size + chunk_size]
            documents_chunks.append(chunk)
            current_size += chunk_size
    return documents_chunks

documents_chunks = fixed_size_chunks(chunk_size=600, overlap=80, documents=pages)

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

# search the index for the top 2 most similar documents to the question
distances, indices = index.search(question_embedding, k=5)
# this returns two arrays: distances that shows similarity scores and 
# indices that shows the positions of the most similar docs in the original doc

# retrieve the actual documents based on the indices returned by FAISS 
similar_docs = [documents_chunks[i] for i in indices[0]]

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

# answer retrieved: 
# According to the provided context, Attention is important for several reasons:

# 1. **Capturing long-range dependencies**: Attention mechanisms like self-attention help capture dependencies between distant positions in a sequence, which is difficult for models to learn.
# 2. **Multi-task learning**: Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, enabling the model to learn multiple tasks at once.
# 3. **Improved model performance**: In tasks like reading comprehension, self-attention has been used successfully, demonstrating its effectiveness in improving model performance.
# 4. **Flexibility and robustness**: Attention-based models can be designed to attend to different parts of the input sequence in different ways, making them more flexible and robust than traditional models.

# Overall, Attention is important because it enables models to effectively process and learn from sequential data by capturing long-range dependencies, learning multiple tasks, and improving model performance.