from flask import Flask, render_template, request, jsonify
from search_func import load_faiss_index, search  # Import search functions
from generate_func import generate_response       # Import generation functions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load FAISS index and embedding model
index = load_faiss_index()  # Load your FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined mental health keywords
def get_keywords():
    return [
        "mental health", "well-being", "schools", "families", "communities",
        "teacher's wellbeing", "school counselors", "special educators",
        "psychosocial support", "risk factors", "childhood conditions",
        "adolescence", "depression", "bullying", "substance abuse",
        "abuse", "trauma", "anxiety", "self-esteem", "trust",
        "attachment concerns", "learning difficulties"
    ]

# Function to check if the query is related to mental health
def is_relevant_query(user_query, keywords):
    return any(keyword in user_query.lower() for keyword in keywords)

# Define route for the chatbot page
@app.route('/')
def chat():
    return render_template('chat.html')

# Define route for generating chatbot responses
@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.form['message']
    keywords = get_keywords()

    # Check if the query contains relevant keywords
    if is_relevant_query(user_query, keywords):
        # Proceed with searching the FAISS index for relevant answers
        results = search(user_query, index, embedding_model)

        # Extract the top retrieved text snippets from FAISS search
        retrieved_texts = [text for _, _, text in results]

        # Generate a response based on the top search results using Cohere
        generated_answer = generate_response(retrieved_texts)
        return jsonify({"response": generated_answer})
    
    else:
        # Respond with a message guiding the user to ask about mental health
        return jsonify({"response": "I am trained on mental health topics. Please ask questions related to mental health. Suggested topics include: depression, anxiety, well-being, trauma, and more."})

if __name__ == '__main__':
    app.run(debug=True)
