import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# App title
st.set_page_config(page_title="🤖 File Q&A Chatbot")

# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx"])

# User-provided prompt
prompt = st.text_input("Type your message here:")

# GPT-2 model parameters
max_length = st.slider('Max response length', min_value=10, max_value=200, value=50)
temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=1.0)

# Function to generate response
def generate_response(prompt):
    input_text = f"User: {prompt}\nAssistant:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate response using GPT-2
    response_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text

# Process the uploaded file when the user uploads a file
if uploaded_file is not None:
    # Read the file into a Pandas DataFrame (adjust this based on your file type)
    df = pd.read_csv(uploaded_file)  # Change this line for different file types

    # Display the content of the uploaded file
    st.write("### Uploaded File Contents:")
    st.write(df)

    # Extract relevant information from the file (modify this based on your file content)
    file_content = " ".join(df.iloc[:, 0].astype(str))  # Assuming the first column contains text data

    # Update the chatbot prompt with file content
    chatbot_prompt = f"Based on the uploaded file, the user is asking: {file_content}"

    # Generate a response using the updated prompt
    response = generate_response(chatbot_prompt)

    # Display the chatbot's response
    st.write("### Assistant's Response")
    st.write(response)

# Generate a response when the user provides a prompt
if st.button("Submit") and prompt:
    # Generate response based on the user's prompt
    response = generate_response(prompt)

    # Display the assistant's response
    st.write("### Assistant's Response")
    st.write(response)
