import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Chat with the Streamlit docs, powered by Hugging Face Transformers", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with the Streamlit docs, powered by Hugging Face Transformers 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-hugging-face-transformers/)", icon="📃")

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# User-provided document
uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf"])

if uploaded_file is not None:
    document = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Document:")
    st.write(document)

    # User-provided prompt
    prompt = st.text_input("Ask a question or request a summary")

    if st.button("Generate Summary"):
        if prompt:
            input_text = f"Document: {document}\nQuestion: {prompt}"
        else:
            input_text = f"Document: {document}"

        # Generate summary
        summary = summarizer(input_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)[0]["summary"]
        st.write("### Summary")
        st.write(summary)
