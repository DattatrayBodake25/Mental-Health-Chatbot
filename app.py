import streamlit as st
from search_func import search
from generate_func import generate_response
import faiss
from sentence_transformers import SentenceTransformer

import streamlit as st

cohere_api_key = st.secrets["cohere_api_key"]


if not cohere_api_key:
    st.error("Cohere API key not found. Please set it in Streamlit Cloud secrets.")
else:
    # Continue with the rest of the app only if the API key is found

    def load_faiss_index():
        """Load the FAISS index from the specified file."""
        index_path = 'mental_health.index'
        index = faiss.read_index(index_path)
        return index

    # Function to get relevant keywords from the content
    def get_keywords():
        # List of keywords extracted from the contents provided
        return [
            "mental health", "well-being", "schools", "families", "communities",
            "teacher's wellbeing", "school counselors", "special educators",
            "psychosocial support", "risk factors", "childhood conditions",
            "adolescence", "depression", "bullying", "substance abuse",
            "abuse", "trauma", "anxiety", "self-esteem", "trust",
            "attachment concerns", "learning difficulties"
        ]

    # Load necessary files
    index = load_faiss_index()  # Load your FAISS index
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Streamlit app structure
    st.title("Mental Health Chat App")

    # User input
    user_query = st.text_input("Ask a question about mental health:")

    # Keywords for redirection
    keywords = get_keywords()

    if st.button("Generate Answer"):
        if user_query:
            # Check if user query contains relevant keywords
            if any(keyword in user_query.lower() for keyword in keywords):
                # Search for relevant text chunks
                results = search(user_query, index, embedding_model)

                # Check if results are found
                if results:
                    # Generate response based on the top results
                    generated_answer = generate_response([text for _, _, text in results])
                    st.subheader("Generated Answer:")
                    st.write(generated_answer)
                else:
                    st.warning("No relevant information found. Please try asking something else.")
            else:
                # Redirect user to mental health topics
                st.warning(f"Please ask a question about mental health. Suggested topics include: {', '.join(keywords)}.")
        else:
            st.warning("Please enter a question.")



# import streamlit as st
# from search_func import search
# from generate_func import generate_response
# import faiss
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import os

# # Load environment variables from the .env file
# load_dotenv()

# # Get the Cohere API key from the environment variable
# cohere_api_key = os.getenv('COHERE_API_KEY')

# if not cohere_api_key:
#     st.error("Cohere API key not found. Please ensure it is set in the .env file.")
# else:
#     # Continue with the rest of the app only if the API key is found

#     def load_faiss_index():
#         """Load the FAISS index from the specified file."""
#         index_path = 'mental_health.index'
#         index = faiss.read_index(index_path)
#         return index

#     # Function to get relevant keywords from the content
#     def get_keywords():
#         # List of keywords extracted from the contents provided
#         return [
#             "mental health", "well-being", "schools", "families", "communities",
#             "teacher's wellbeing", "school counselors", "special educators",
#             "psychosocial support", "risk factors", "childhood conditions",
#             "adolescence", "depression", "bullying", "substance abuse",
#             "abuse", "trauma", "anxiety", "self-esteem", "trust",
#             "attachment concerns", "learning difficulties"
#         ]

#     # Load necessary files
#     index = load_faiss_index()  # Load your FAISS index
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Streamlit app structure
#     st.title("Mental Health Chat App")

#     # User input
#     user_query = st.text_input("Ask a question about mental health:")

#     # Keywords for redirection
#     keywords = get_keywords()

#     if st.button("Generate Answer"):
#         if user_query:
#             # Check if user query contains relevant keywords
#             if any(keyword in user_query.lower() for keyword in keywords):
#                 # Search for relevant text chunks
#                 results = search(user_query, index, embedding_model)

#                 # Check if results are found
#                 if results:
#                     # Generate response based on the top results
#                     generated_answer = generate_response([text for _, _, text in results])
#                     st.subheader("Generated Answer:")
#                     st.write(generated_answer)
#                 else:
#                     st.warning("No relevant information found. Please try asking something else.")
#             else:
#                 # Redirect user to mental health topics
#                 st.warning(f"Please ask a question about mental health. Suggested topics include: {', '.join(keywords)}.")
#         else:
#             st.warning("Please enter a question.")
