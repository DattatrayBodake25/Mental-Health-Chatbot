import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize your embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model from Sentence Transformers

def load_text_chunks(file_path):
    """Load text chunks from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text_chunks = f.read().splitlines()  # Split file into lines
    return text_chunks

# Load the original text chunks
original_text_chunks = load_text_chunks('mental_health_chunks.txt')  # Path to your text chunks file

def load_faiss_index():
    """Load the FAISS index from a file."""
    index = faiss.read_index("mental_health.index")  # Path to your FAISS index file
    return index

def search(query, index, embedding_model, k=5):
    """Search for the top k results in the FAISS index based on the user's query."""
    query_vector = embedding_model.encode(query).astype('float32')  # Encode the query

    # Perform the search
    distances, indices = index.search(np.array([query_vector]), k)

    results = []
    for i in range(k):
        idx = indices[0][i]
        distance = distances[0][i]
        if idx < len(original_text_chunks):  # Check index bounds
            results.append((idx, distance, original_text_chunks[idx]))  # Append result

    return results