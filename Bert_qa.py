import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load a suitable model from Hugging Face Transformers
model_name = "facebook/bart-base"  # You can choose a different model if needed
tokenizer = AutoTokenizer.from_pretrained(bert-base-uncased)
model = AutoModelForSeq2SeqLM.from_pretrained(bert-base-uncased)

st.title(" File Q&A with Hugging Face Transformers")

uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    article = uploaded_file.read().decode()
    prompt = f"""Question: {question}\nContext: {article}"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    st.write("### Answer")
    st.write(answer)
