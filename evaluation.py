from phase06 import (load_model_chunker, load_and_chunk, get_bm25_score, get_doc_semantic_index, run_pipeline, retrieve_sources)
from pathlib import Path

model_name = "sentence-transformers/all-mpnet-base-v2" 
directory = Path.cwd() /"uploads"


splitter, crossEncoder = load_model_chunker(model_name)
metadata, document_chunks = load_and_chunk(directory, splitter)
bm25 = get_bm25_score(document_chunks)
index, model = get_doc_semantic_index(document_chunks)


valid_sources = ["attention.pdf", "security.pdf", "healthcare.pdf"]


EVAL_SET = [
    ("What real-world data breach example is cited to illustrate data leakage risk?",
     "The Equifax data breach, which caused substantial legal liabilities and damaged public trust.",
     "security.pdf"),

    ("What is an adversarial sample attack in image recognition?",
     "Attackers add small, imperceptible perturbations to an image, causing the model to misjudge it.",
     "security.pdf"),

    ("What causes algorithm bias in recruitment AI systems?",
     "Historic tendencies in training data -- favoritism toward specific genders, ethnicities, or credentials.",
     "security.pdf"),

    ("What are RBAC and ABAC?",
     "RBAC assigns permissions to roles then roles to users; ABAC dynamically evaluates attributes to authorize access.",
     "security.pdf"),

    ("What four technical means does the security paper propose for enhancing algorithm robustness?",
     "Adversarial training, model fusion, visualization technology, and rule extraction.",
     "security.pdf"),

    ("What are the five main application areas of AI in healthcare?",
     "Diagnostic assistance, treatment personalization, patient monitoring and care, healthcare operations, and public health/epidemiology.",
     "healthcare.pdf"),

    ("What five major challenges does AI face in healthcare integration?",
     "Data privacy and security, ethical and legal considerations, interoperability, scalability and accessibility, and human-AI interaction.",
     "healthcare.pdf"),

    ("Which two data protection regulations must healthcare AI providers comply with?",
     "GDPR (Europe) and HIPAA (United States).",
     "healthcare.pdf"),

    ("How does AI assist in drug development?",
     "AI predicts how chemical compounds interact with biological targets, speeding up discovery and reducing cost.",
     "healthcare.pdf"),

    ("What role should AI play relative to healthcare professionals' judgment?",
     "AI should support decision-making rather than replace human judgment, with human oversight maintained.",
     "healthcare.pdf"),
      ("What is self-attention?",
     "A mechanism that relates different positions of a single sequence to compute a representation of that sequence.",
     "attention.pdf"),

    ("What two sub-layers does each encoder layer contain?",
     "A multi-head self-attention mechanism and a position-wise fully connected feed-forward network.",
     "attention.pdf"),

    ("How does the decoder prevent positions from attending to subsequent positions?",
     "By masking out (setting to negative infinity) values in the scaled dot-product attention corresponding to future positions.",
     "attention.pdf"),

]

def evaluate(k=3, verbose=True):
    hits = 0
    for query, expected_answer, expected_doc in EVAL_SET:
        retrieved = retrieve_sources(query, valid_sources, document_chunks, metadata, bm25, index, model, k, crossEncoder)
        hit = expected_doc in retrieved
        hits += hit
        if verbose:
            mark = "Sucess" if hit else "Failure"
            print(f"{mark}  {query[:60]:<60} expected={expected_doc:<15} got={retrieved}")
    score= hits / len(EVAL_SET)
    print(f"\nRecall@{k}: {score:.2f}  ({hits}/{len(EVAL_SET)})")
    return score


if __name__ == "__main__":
    for k in (1, 3, 5):
        print(f"\n=== K={k} ===")
        evaluate(k=k)


