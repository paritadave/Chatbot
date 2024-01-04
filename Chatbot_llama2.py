import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# App title
st.set_page_config(page_title="🦙💬 Chatbot Demo")

# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx"])

# Store generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hey, How may I assist you?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hey, How may I assist you?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Process the uploaded file when the user drops a file
if uploaded_file is not None:
    # Read the file into a string (adjust this based on your file type)
    file_content = uploaded_file.read().decode()

    # User-provided prompt
    prompt = st.text_input("Ask something about the file", placeholder="e.g., Can you summarize the file?", key="user_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Use a pre-trained question-answering model from Hugging Face Transformers
        model_name = "distilbert-base-cased-distilled-squad"  # You can choose a different model if needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        # Tokenize the input
        inputs = tokenizer(prompt, file_content, return_tensors="pt")

        # Get the model's prediction
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        # Display the chatbot's response
        response = f"Assistant: {answer}"
        with st.chat_message("assistant"):
            st.write(response)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # For simplicity, let's just echo the prompt back as the response
            response = f"Assistant: {prompt}"
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
