from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the E5 model and tokenizer
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def generate_embedding(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence embeddings
    return embeddings.numpy()

# Sample corpus and query
corpus = [
    "Varför springer tigrar så fort?",
    "Jag är sugen på kladdkaka",
    "Det vore ballt att bli bättre på golf."
]
query = "Vad kan jag göra för att förbättra mina golffärdigheter?"

# Generate embeddings for the corpus and query
corpus_embeddings = generate_embedding(corpus)
query_embedding = generate_embedding([query])

# Perform retrieval using cosine similarity
similarities = cosine_similarity(query_embedding, corpus_embeddings)
most_similar_index = np.argmax(similarities)
retrieved_doc = corpus[most_similar_index]

# Output the most relevant document
print(f"Query: {query}")
print(f"Retrieved Document: {retrieved_doc}")