# ## Phase 1 — Build the Minimal RAG (50 lines)

# **Task:** Create a small document list (5–10 sentences on any topic you know well), 
# embed them, store in an index,
#  retrieve the top-2 most similar to a query,
#  and pass those to an LLM to generate an answer.

# ### Sub-challenges:

# **1a. Embedding**
# > **Hint:** `SentenceTransformer` takes a model name string. 
# The model used in the tutorial is lightweight (~22MB). Call `.encode()` on a list of strings — 
# what does the output shape look like? What does each row represent?

# **1b. Indexing**
# > **Hint:** FAISS needs to know the *dimensionality* of your vectors before you add anything.
#  That dimension comes from the embedding model. `IndexFlatL2` uses Euclidean distance — 
# think about what that means geometrically for similarity.

# **1c. Retrieval**
# > **Hint:** You encode the *query* the same way you encoded the documents. 
# Then `.search(query_embedding, k)` returns two arrays — 
# what do they represent? How do you map indices back to your original document strings?

# **1d. Generation**
# > **Hint:** Build the prompt as a Python f-string. 
# The key prompt engineering insight here: 
# you want the LLM to answer *only from the context*, not from its parametric memory. How would you phrase that instruction?

# **Stop point:** Test with 3 different queries. 
# Does retrieval feel right? If not, why might it be wrong before you even touch the LLM?

# ---
# A list of strings detailing a standard corporate procedure for handling customer complaints
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import os 
from dotenv import load_dotenv 

# prepare the docs
documents = [
    "Initial Intake and Logging: The customer support representative receives the complaint via email, phone, or chat and immediately logs all relevant details into the company's central CRM database to ensure traceability.",
    "Triage and Classification: A specialized team reviews the ticket within two hours of submission to categorize the issue by severity, assigning a priority level that dictates how quickly it must be resolved.",
    "Investigation and Root Cause Analysis: The assigned specialist investigates the root cause of the problem by analyzing technical logs, interviewing relevant staff, or reviewing account histories to prevent similar issues from happening to other customers.",
    "Resolution Proposal and Communication: Once a solution is identified, the specialist contacts the customer to present the proposed resolution, which may include a technical fix, a service credit, or a product replacement depending on company policy.",
    "Quality Assurance and Case Closure: Before the ticket is permanently closed, a quality assurance manager follows up with the customer to ensure they are fully satisfied with the outcome, officially archiving the case only after confirmation is received."
]

# embed the documents
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
document_embeddings = model.encode(documents)

# store the embeddings in a FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# prepare the question 
question = "According to the customer complaint handling procedure:1. What is the first step in handling a customer complaint. 2. How are complaints prioritized? 3. Who verifies customer satisfaction before a case is closed?"

#embed the question
question_embedding = model.encode([question])

# search the index for the top 2 most similar documents to the question
distances, indices = index.search(question_embedding, k=4)
# this returns two arrays: distances that shows similarity scores and 
# indices that shows the positions of the most similar docs in the original doc

# retrieve the actual documents based on the indices returned by FAISS 
similar_docs = [documents[i] for i in indices[0]]

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