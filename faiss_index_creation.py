from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Load the pre-trained model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your mental health chunks
with open("mental_health_chunks.txt", "r", encoding='utf-8') as f:
    documents = f.readlines()

# Create embeddings for all chunks
embeddings = model.encode(documents, convert_to_tensor=True)

# Convert embeddings to a numpy array
embeddings_np = embeddings.cpu().detach().numpy()

# Create FAISS index and add embeddings
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

# Save the FAISS index
faiss.write_index(index, "mental_health.index")
print("FAISS index created and saved as 'mental_health.index'")
