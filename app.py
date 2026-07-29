import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# COLORS FIRST
my_custom_css = """
/* Background */
body, gradio-app, .main, .gradio-container {
    background: linear-gradient(180deg, #EAF7E5, #D8F3DC);
    font-family: Arial, Helvetica, sans-serif;
}

/* Main container */
.gradio-container {
    max-width: 950px !important;
    margin: auto;
}

/* User bubble */
.user-row .message,
div[data-testid="user-message"]{
    background:#A8E6A2 !important;
    color:#173B2D !important;
    border-radius:18px !important;
    padding:14px !important;
}

/* Bot bubble */
.bot-row .message,
div[data-testid="bot-message"]{
    background:white !important;
    color:#173B2D !important;
    border-radius:18px !important;
    padding:14px !important;
    border:2px solid #D7EED2;
}

/* Textbox */
textarea{
    border-radius:15px !important;
}

/* Send button */
button.primary{
    background:#2E8B57 !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
}

button.primary:hover{
    background:#256D46 !important;
}

/* Titles */
h1{
    color:#1E5631;
    font-size:42px;
}

h2{
    color:#2E8B57;
}

h3{
    color:#3B7A57;
}

p{
    color:#264D35;
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
    system_message = f"""
    You are GreenLAKES, an environmental chatbot.

    Use the provided information to answer questions about:
    - pollution
    - sustainability
    - recycling
    - climate issues
    - local environmental actions

    Give helpful and clear answers.
    Encourage users that individual actions can contribute to larger environmental change.

    Information:
    {rag_info}
    """
    
    messages = [{"role": "system", "content": system_message}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
    )

    return response.choices[0].message.content.strip()

with gr.Blocks() as demo:

    gr.HTML("""
    <center>

    <img src="/file=logo.png" width="180">

    <h1>GreenLAKES</h1>

    <p>
    Educating users about pollution and providing strategies
    for sustainable action in their communities.
    </p>

    </center>
    """)


    gr.Markdown("""
## 🌱 GreenLAKES

Ask questions about:

♻️ Plastic Pollution

💨 Air Pollution

🖥️ E-Waste

🌎 Climate Change

🏡 Sustainable Habits

""")



    gr.HTML("""
    <div style="text-align:center">

    <h3>
    Explore ways to reduce your environmental impact:
    </h3>

    </div>
    """)


    gr.ChatInterface(
        fn=respond
    )


    gr.Markdown("""
    ### Try asking:

    How can I reduce plastic waste?

    What is e-waste?

    How does pollution affect the environment?

    Why is air pollution harmful?

    What actions can I take to help the environment?

""")

    demo.launch(css=my_custom_css)

# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!