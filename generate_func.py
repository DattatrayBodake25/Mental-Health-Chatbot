import cohere
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Cohere client
cohere_api_key = os.getenv('COHERE_API_KEY')  # Retrieve the API key from .env
co = cohere.Client(cohere_api_key)

def generate_response(retrieved_texts):
    # Combine retrieved texts into a single prompt
    prompt = "Based on the following information, provide a concise response within 150 tokens:\n\n" + "\n".join(retrieved_texts)
    
    # Generate response using Cohere
    response = co.generate(
        model='command',  # You can choose the model size as per your requirement
        prompt=prompt,
        max_tokens=400,  # Limit to 400 tokens
        temperature=0.5  # Adjust for creativity
    )
    
    return response.generations[0].text.strip()










# import cohere
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Initialize Cohere client
# cohere_api_key = os.getenv('COHERE_API_KEY')  # Retrieve the API key from .env
# co = cohere.Client(cohere_api_key)

# def generate_response(retrieved_texts):
#     # Combine retrieved texts into a single prompt
#     prompt = "Based on the following information, provide a concise response within 150 tokens:\n\n" + "\n".join(retrieved_texts)
    
#     # Generate response using Cohere
#     response = co.generate(
#         model='command',  # You can choose the model size as per your requirement
#         prompt=prompt,
#         max_tokens=400,  # Limit to 150 tokens
#         temperature=0.5  # Adjust for creativity
#     )
    
#     return response.generations[0].text.strip()






# # import cohere

# # # Initialize Cohere client
# # cohere_api_key = 'my-cohere-key'  # Replace with your Cohere API key
# # co = cohere.Client(cohere_api_key)

# # def generate_response(retrieved_texts):
# #     # Combine retrieved texts into a single prompt
# #     prompt = "Based on the following information, provide a concise response within 150 tokens:\n\n" + "\n".join(retrieved_texts)
    
# #     # Generate response using Cohere
# #     response = co.generate(
# #         model='command',  # You can choose the model size as per your requirement
# #         prompt=prompt,
# #         max_tokens=400,  # Limit to 150 tokens
# #         temperature=0.5  # Adjust for creativity
# #     )
    
# #     return response.generations[0].text.strip()