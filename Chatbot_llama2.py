import streamlit as st
import pandas as pd

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot Demo")

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
    # Read the file into a Pandas DataFrame (adjust this based on your file type)
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded File Contents:")
    st.write(df)

    # User-provided prompt
    prompt = st.text_input("Ask something about the file", placeholder="e.g., Can you summarize the file?", key="user_prompt")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Extract relevant information from the file based on your logic
        # You may use natural language processing, custom logic, or other libraries here
        # For simplicity, let's just echo the prompt back as the response
        response = f"Assistant: {prompt}"

        # Display the chatbot's response
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
