import streamlit as st
import replicate
import os
import pandas as pd

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot Demo")

# File upload widget
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx"])

with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot Demo')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    
    #st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
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
def generate_llama2_response(llm, prompt_input):
    # Update the input format based on the Replicate API requirements
    input_data = {"inputs": [prompt_input], "system_prompt": "You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."}
    output = replicate.run(llm, input_data)
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
