# [GreenLAKES]

This chatbot is intended to gives you simple and location-based solutions to combat pollution and helps you learn more about protecting the environment!

🤗 **Originally built as a Hugging Face Space:** [https://huggingface.co/spaces/kode-with-klossy/3.3-groupD2-capstone]

> ⚠️ Note: This Space is no longer live. The code in this repo is the full project.

!Screenshot of my chatbot(<img width="783" height="657" alt="image" src="https://github.com/user-attachments/assets/95299db5-657e-4cf2-a25e-5aea4b00f192" />)

## What it does

- Individuals can ask the chatbot personalized questions about any type of pollution, including the 5 main types and lesser stressed issues like space debris. Prompts can be inputted using either the standard text option or through voice input, and if a user is unable to think of a prompt, FAQs are provided in a panel on the right for the user to click on to prompt the chatbot.
- Individuals can generate a daily eco-tip, which is randomly generated and covers several sectors of sustainability.
- Individuals can calculate how much water consumption they mitigate based on their reusable water usage or shower duration.
- Individuals can receive personalized tips on sustainable waste management across several sectors based on their location (city, state).

## How it works

When a user types a message, the chatbot pulls information through semantic search through its extensive knowledge in its main library, in the format of a text file. This research compiled across the team covers all major aspects of sustainability, including minor actions one can take to practice environmentally conscious habits, as well as thorough information about all types of pollution. Additionally, if the user asks for links to learn more about a specific topic, the chatbot pulls specific links matching the description from its library as well.

## Built with

- **Gradio** — the interface
- **Hugging Face Inference Providers** — the AI model ([Qwen/Qwen2.5-7B-Instruct])
- **Sentence Transformers** — vector embeddings for knowledge base search (all-MiniLM-L6-v2)
- **Faster-Whispers** – speech-to-text transcription for voice chat (base model)

## What I learned

The hardest part of building GreenLAKES was getting custom UI components—like the dark mode toggle and leaf background image—to play nicely with Gradio's default styling framework. 

* **Executing Dynamic JavaScript in Gradio:** Standard `<script>` tags don't run reliably inside Gradio's `gr.HTML` component because the DOM updates dynamically. I solved this by moving the light/dark theme switching logic directly into inline `onclick` event attributes on the HTML toggle button itself.
* **Managing CSS Opacity Across Themes:** Gradio applies solid background fills to containers by default, which completely covered our jungle background image. I fixed this by overriding container backgrounds with `transparent !important` and using semi-transparent `rgba()` color variables for light and dark modes, allowing the background leaves to remain visible across both themes.
* **Connecting a Multimodal RAG Pipeline:** Combining `SentenceTransformers` vector search, `Faster-Whisper` voice transcription, and `Qwen2.5-7B-Instruct` required balancing retrieval context with concise system prompts so the bot remains fast, accurate, and conversational.

## About

Built at Kode With Klossy (https://www.kodewithklossy.com) AI/ML Camp,
Summer 2026, by Sharanya Vats, Elisa Reyes, Anastasiia Romanenko, Kaelyn Neiswonger, Loza Girma.
