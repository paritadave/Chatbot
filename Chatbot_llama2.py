import streamlit as st
import replicate
import os
import pandas as pd

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot Demo")

# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx"])

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot Demo')
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    # ... (your existing sidebar code)


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hey, How may I assist you?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hey, How may I assist you?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(llm, prompt_input):
    output = replicate.run(llm,
                           input={"prompt": f"{prompt_input} Assistant: ",
                                  "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
    return output

# Process the uploaded file when the user drops a file
if uploaded_file is not None:
    # Read the file into a Pandas DataFrame (adjust this based on your file type)
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded File Contents:")
    st.write(df)

    # Extract relevant information from the file
    file_content = df.to_string(index=False)  # Use the DataFrame content as the file content

    # User-provided prompt
    prompt = st.text_input("Ask something about the file", placeholder="e.g., Can you summarize the file?", key="user_prompt")
    
    # Define llm here based on your model selection logic
    llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'  # Update this line with your logic for selecting the model
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Update the chatbot prompt with file content
        chatbot_prompt = f"Based on the uploaded file, the user is asking: {file_content}"

        # Generate a new response using the updated prompt
        response = generate_llama2_response(llm, chatbot_prompt)

        # Display the chatbot's response
        with st.chat_message("assistant"):
            st.write(response)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(llm, prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
