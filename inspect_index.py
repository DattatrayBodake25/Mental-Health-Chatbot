import faiss
import numpy as np

# Load the FAISS index
index = faiss.read_index("mental_health.index")

# Load your original text chunks from the mental_health_chunks.txt file
with open("mental_health_chunks.txt", "r", encoding="utf-8") as file:
    original_chunks = file.readlines()

# Check how many vectors are in the index
num_vectors = index.ntotal

print(f"Total number of vectors in the index: {num_vectors}")

# Print the first few vectors and their corresponding text chunks
for i in range(min(num_vectors, 5)):  # Adjust number as needed
    print(f"Vector {i}: {index.reconstruct(i)}")  # This will print the raw vector
    print(f"Original Text Chunk {i}: {original_chunks[i].strip()}\n")
