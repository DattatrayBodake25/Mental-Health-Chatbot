import faiss

# Load the FAISS index
index = faiss.read_index("mental_health.index")

# Print the number of vectors in the index
print("Number of vectors in the index:", index.ntotal)

# Print the dimensionality of the vectors
print("Dimension of each vector:", index.d)
