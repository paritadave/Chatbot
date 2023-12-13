import streamlit as st
import replicate
import os
import pandas as pd

# App title
st.set_page_config(page_title="🦙💬 File Q&A with Llama 2")

# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx"])

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 File Q&A with Llama 2')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

os.environ['REPLICATE_API_TOKEN'] = replicate_api

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
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
    return output

# Process the uploaded file when the user drops a file
if uploaded_file is not None:
    # Read the file into a Pandas DataFrame (adjust this based on your file type)
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "text/plain":
        df = pd.read_csv(StringIO(uploaded_file.getvalue()), sep="\t")
    # Add more conditions for other file types if needed

    st.write("Uploaded File Contents:")
    st.write(df)

    # Extract relevant information from the file
    file_content = uploaded_file.getvalue()

    # Update the chatbot prompt with file content
    chatbot_prompt = f"Based on the uploaded file, the user is asking: {file_content}"

    # Generate a new response using the updated prompt
    response = generate_llama2_response(chatbot_prompt)

    # Display the chatbot's response
    with st.chat_message("assistant"):
        st.write(response)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
