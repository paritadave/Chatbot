import streamlit as st
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the LLAMA 2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/lm-adapter-text-davinci-001")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/lm-adapter-text-davinci-001")

# Define a function to generate text using the LLAMA 2 model
def generate_text(prompt):
    # Encode the prompt into tokens
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text using the LLAMA 2 model
    outputs = model.generate(input_ids=input_ids, max_length=100)

    # Decode the outputs into text
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_text

# Define a function to handle user input
def handle_user_input(user_input):
    # Generate text using the LLAMA 2 model
    bot_response = generate_text(user_input)

    # Return the bot response
    return bot_response

# Create a Streamlit app
st.title("LLAMA 2 Trial Bot")

# Get the user input
user_input = st.text_input("Enter your question:")

# Generate a response using the LLAMA 2 model
bot_response = handle_user_input(user_input)

# Display the bot response
st.write(bot_response)
