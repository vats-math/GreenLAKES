import gradio as gr
import random
from faster_whisper import WhisperModel
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# COLORS FIRST
my_custom_css = """
/* Light theme */
[data-theme="light"] {
    --bg: #DDEFD0;
    --text: #173B2D;
    --card: rgba(255,255,255,.88);
}

/* Dark theme */
[data-theme="dark"] {
    --bg: #143322;
    --text: #EAF7E5;
    --card: rgba(20,50,35,.88);
}

body {
    background: var(--bg);
    color: var(--text);
}

.card {
    background: var(--card);
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
/* Dark Background */
.dark body, .dark gradio-app, .dark .main, .dark .gradio-container {
    background: linear-gradient(180deg, #122119, #0B1510) !important;
}

/* Dark User Bubble */
.dark .user-row .message, .dark div[data-testid="user-message"] {
    background: #23533E !important;
    color: #E2F4DF !important;
}

/* Dark Bot Bubble */
.dark .bot-row .message, .dark div[data-testid="bot-message"] {
    background: #192D23 !important;
    color: #E2F4DF !important;
    border: 2px solid #2D4D3D !important;
}

/* Dark Input & Textarea */
.dark textarea, .dark input {
    background-color: #16281F !important;
    color: #E2F4DF !important;
    border: 1px solid #2D4D3D !important;
}

/* Dark Send Button */
.dark button.primary {
    background: #41AA70 !important;
    color: #0B1510 !important;
    font-weight: bold;
}
.dark button.primary:hover {
    background: #52C284 !important;
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
# Load text to speech model
whisper = WhisperModel("base")

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

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")


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
    Answer questions concisely. Do not repeat greeting introductions if the user just says hello.
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


# VOICE CHAT
def transcribe(audio_file):
    """
    Converts microphone recording
    into text.
    """

    segments, info = whisper.transcribe(
        audio_file
    )


    text = ""

    for segment in segments:
        text += segment.text


    return text.strip()

def voice_chat(audio, history):
    print("Audio received:", audio)

    if audio is None:
        return history

    # Convert speech to text
    user_text = transcribe(audio)

    print("Transcribed:", user_text)

    # Get chatbot response
    response = respond(user_text, history)

    # Add message to chat history
    history.append({
        "role": "user",
        "content": user_text
    })

    history.append({
        "role": "assistant",
        "content": response
    })

    return history


def text_chat(message, history):
    """
    Handles normal typing.
    """

    answer = respond(
        message,
        history
    )


    history = history + [
        {
            "role": "user",
            "content": message
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]


    return history, history, ""


# eco-friendly tips

ECO_TIPS = [
    "💡 **Energy:** Unplug electronics like TVs, gaming consoles, and microwave clocks when not in use to prevent 'phantom energy' draw.",
    "💧 **Water:** Turning off the tap while brushing your teeth can save up to 8 gallons of water a day!",
    "🛍️ **Shopping:** Keep reusable grocery bags in your car, backpack, or near your front door so you never forget them.",
    "🥗 **Food:** Try 'Meatless Mondays'—eating plant-based meals even one day a week significantly shrinks your carbon footprint.",
    "📱 **E-Waste:** Store old phones and chargers in a box and drop them off at certified e-waste recycling bins instead of throwing them away.",
    "🚿 **Water:** Cutting just 2 minutes off your shower time saves up to 5 gallons of water every single time!",
    "🧺 **Laundry:** Washing your clothes in cold water saves around 90% of the energy your washing machine uses to heat water.",
    "🚲 **Commute:** Walking, biking, or taking public transit for short trips cuts down on greenhouse gas emissions and vehicle wear-and-tear."
]

def get_random_tip():
    return random.choice(ECO_TIPS)


def calculate_impact(plastic_bottles, shower_mins):
    bottles_saved_year = plastic_bottles * 52
    water_saved_year = shower_mins * 2.5 * 365

    return f"""
    ###  Your Estimated Annual Impact:
    * 🧋 **Single-Use Bottles Saved:** ~**{bottles_saved_year:,}** plastic bottles / year
    * 💧 **Water Saved:** ~**{water_saved_year:,.0f}** gallons of water / year
    
    *Every small daily habit adds up to massive environmental change!*
    """

def find_local_resources(location, category):
    category_names = {
        "e-waste": "E-Waste and Electronics (laptops, phones, batteries)",
        "compost": "Organic scrap and composting facilities",
        "hazmat": "Household Hazardous Waste (paint, chemicals, oil)",
        "plastics": "Plastics and household recycling bins"
    }
    
    cat_label = category_names.get(category, category)
    loc_text = location.strip() if location and location.strip() else "general municipal areas"

    prompt = f"""
    You are an environmental assistant. The user is asking where to dispose of {cat_label} in or near: {loc_text}.
    
    Provide specific, helpful guidance for {loc_text}:
    1. **Disposal Guidelines**: 2 short rules on how to prepare these materials.
    2. **Recommended Local Drop-Off Centers**: List 3 specific local options, drop-off hubs, or major retailer programs available in or near {loc_text}.
    3. **Pro Tip**: One actionable safety or efficiency tip.

    Do not use emojis. Keep the formatting clean with bold text and bullet points.
    """

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


with gr.Blocks(css=my_custom_css) as demo:


    gr.HTML("""
    <center>
    <img src="https://huggingface.co/spaces/kode-with-klossy/3.3-groupD2-capstone/resolve/main/logo.png" alt="logo.png" width="180">
    <h1>GreenLAKES</h1>
    <p>
    Educating users about pollution and providing strategies
    for sustainable action in their communities.
    </p>
    </center>
    """)

 
    with gr.Tabs():

        # tab 1 chat bot
        with gr.Tab("💬 Chat Assistant"):
            with gr.Row():
                
            
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        value=[{
                            "role": "assistant",
                            "content": "Hello! I am GreenLAKES. How can I help you today with questions about pollution, recycling, or sustainability?"
                        }],
                        height=520
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask a question...",
                            show_label=False,
                            scale=4
                        )
                        submit_btn = gr.Button("Send Text", variant="primary", scale=1)
                        
                    with gr.Row():
                        # CREATE MIC BUTTON
                        microphone = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record Voice",
                            scale=4
                        )
                        mic_button = gr.Button("🎤 Send Voice", scale=1)
            
                with gr.Column(scale=2):
                    gr.Markdown("### 🌿 Daily Eco-Habits & FAQs")
                    
                    faq_1 = gr.Button("How can I reduce plastic waste?")
                    faq_2 = gr.Button("How can I save energy at home?")
                    faq_3 = gr.Button("What food scraps can I compost at home?")
                    faq_4 = gr.Button("How do I safely dispose of e-waste?")
                    faq_5 = gr.Button("What habits help reduce food waste?")
                    faq_6 = gr.Button("What daily habits help conserve water?")
                    faq_7 = gr.Button("How can I make my daily commute greener?")
                    faq_8 = gr.Button("How can I shop for clothes sustainably?")


            msg.submit(text_chat, inputs=[msg, chatbot], outputs=[chatbot, chatbot, msg])
            submit_btn.click(text_chat, inputs=[msg, chatbot], outputs=[chatbot, chatbot, msg])
            

            faq_buttons = [faq_1, faq_2, faq_3, faq_4, faq_5, faq_6, faq_7, faq_8]
            for btn in faq_buttons:
                btn.click(text_chat, inputs=[btn, chatbot], outputs=[chatbot, chatbot, msg])

            #VOICE CHAT
            mic_button.click(
                voice_chat,
                inputs=[microphone, chatbot],
                outputs=[chatbot]
            )

        # tab 2 random daily tips
        with gr.Tab("🌱 Daily Eco-Tip Generator"):
            gr.Markdown("### 🎲 Get a Quick Eco-Friendly Habit Tip")
            gr.Markdown("Click the button below to generate a simple, actionable eco-tip you can try today!")
            
            tip_display = gr.Markdown("Click 'Generate Eco-Tip' to get started!")
            generate_tip_btn = gr.Button("Generate Eco-Tip 🎲", variant="primary")

    
            generate_tip_btn.click(get_random_tip, outputs=tip_display)

        #tab 3 impact tab

        with gr.Tab("📊 Impact Calculator"):
            gr.Markdown("### Personal Impact Calculator")
            gr.Markdown("Adjust the sliders below to see how much water and plastic waste your daily habits can save in a year!")
            
            with gr.Row():
                with gr.Column():
                    slider_bottles = gr.Slider(
                        minimum=0, maximum=30, value=7, step=1,
                        label="Reusable water bottles used per week (instead of single-use)"
                    )
                    slider_shower = gr.Slider(
                        minimum=0, maximum=15, value=3, step=1,
                        label="Minutes reduced from daily shower time"
                    )
                    calc_btn = gr.Button("Calculate Impact", variant="primary")
                
                with gr.Column():
                    results_display = gr.Markdown("Adjust sliders and click **Calculate Impact** to see your results!")

            calc_btn.click(
                calculate_impact,
                inputs=[slider_bottles, slider_shower],
                outputs=results_display
            )

        # tab 4 local recycling 
        with gr.Tab("🏠 Local Eco-Finder"):
            gr.Markdown("### Local Recycling & Disposal Finder")
            gr.Markdown("Find out where to safely dispose of specialized waste like electronics, hazardous items, or compost in your community.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    location_input = gr.Textbox(
                        label="Your City and Country",
                        placeholder="e.g. Warsaw Poland"
                    )
                    category_input = gr.Radio(
                        choices=[
                            ("E-Waste (Phones, Laptops, Batteries)", "e-waste"),
                            ("Food Scraps & Compost", "compost"),
                            ("Hazardous Waste (Paint, Chemicals)", "hazmat"),
                            ("Plastics, Glass & Household Recyclables", "plastics")
                        ],
                        value="e-waste",
                        label="What item do you need to dispose of?"
                    )
                    search_btn = gr.Button("Find Local Guidance", variant="primary")
                
                with gr.Column(scale=2):
                    finder_output = gr.Markdown("Select a category and click **Find Local Guidance** to view instructions and drop-off hubs.")

            search_btn.click(
                find_local_resources,
                inputs=[location_input, category_input],
                outputs=finder_output
            )

demo.launch(css=my_custom_css)

