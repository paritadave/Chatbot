import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# App title
st.set_page_config(page_title="🤖 Document Q&A Demo")

# File upload widget
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

# GPT-2 model parameters
max_length = st.slider('Max response length', min_value=10, max_value=200, value=50)
temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=1.0)

# Function to generate response
def generate_response(document_text):
    input_text = f"Document: {document_text}\nQuestion:"
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

# Process the uploaded file when the user drops a file
if uploaded_file is not None:
    # Read the file content
    document_text = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Document Contents:")
    st.write(document_text)

    # Generate response based on the document
    response = generate_response(document_text)

    # Display the chatbot's response
    st.write("### Answer")
    st.write(response)
