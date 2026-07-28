import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# COLORS FIRST
my_custom_css = """
body, gradio-app, .main, div.gradio-container {
    background-color: #FFDAA2 !important;
}
.user-row .message, div[data-testid="user-message"] {
    background-color: #A8FF70 !important;
    color: #092E00 !important;
}
.bot-row .message, div[data-testid="bot-message"] {
    background-color: #FFB8E7 !important;
    color: #092E00 !important;
}
button#submit-btn, button.primary {
    background-color: #00883E !important;
    color: white !important;
}
"""

# Open the knowledge.txt file in read mode with UTF-8 encoding
with open("knowledge.txt", "r", encoding="utf-8") as file:
  # Read the entire contents of the file and store it in a variable
  knowledge_base = file.read()

# Print the text below
print(knowledge_base)

def preprocess_text(text):
  # Strip extra whitespace from the beginning and the end of the text
  cleaned_text = text.strip()

  # Split the cleaned_text by every newline character (\n)
  chunks = cleaned_text.split("\n")

  # Create an empty list to store cleaned chunks
  cleaned_chunks = []

  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  for chunk in chunks:
    stripped_chunk = chunk.strip()
    cleaned_chunks.append(stripped_chunk)

  # Print cleaned_chunks

  print(cleaned_chunks)
  # Print the length of cleaned_chunks
  print(len(cleaned_chunks))

  # Return the cleaned_chunks
  return cleaned_chunks

# Call the preprocess_text function and store the result in a cleaned_chunks variable
cleaned_chunks = preprocess_text(knowledge_base) # Complete this line

# Load the pre-trained embedding model that converts text to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the text_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)

  # Return the chunk_embeddings
  return chunk_embeddings

# Call the create_embeddings function and store the result in a new chunk_embeddings variable
chunk_embeddings = create_embeddings(cleaned_chunks) # Complete this line

# Define a function to find the most relevant text chunks for a given query, chunk_embeddings, and text_chunks
def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  for i in top_indices:
    relevant_info = text_chunks[i]
    top_chunks.append(relevant_info)
  # Return the list of most relevant chunks
  return top_chunks

# Call the get_top_chunks function with the original query
top_results = get_top_chunks("Where does water go after it rains?", chunk_embeddings, cleaned_chunks) # Complete this line

# Print the top results
print(top_results)

# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")


def respond(message, history):

    rag_info = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    system_message = f"You are a friendly chatbot who uses {rag_info} to answer questions about Kode with Klossy."
    
    messages = [{"role": "system", "content": system_message}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()

chatbot = gr.ChatInterface(fn=respond, css=my_custom_css)

chatbot.launch()


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!

import gradio as gr


# 2. USE IT IN YOUR BLOCKS AFTER IT'S DEFINED
with gr.Blocks(css=my_custom_css) as demo:
    # Your HTML header / logo / ChatInterface code goes inside here!
    pass