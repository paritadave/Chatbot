import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load a suitable model from Hugging Face Transformers
model_name = "facebook/bart-base"  # You can choose a different model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.title("File Q&A with Hugging Face Transformers")

uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

try:
    if uploaded_file and question:
        article = uploaded_file.read().decode()
        prompt = f"""Question: {question}\nContext: {article}"""

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Check if the input exceeds the model's maximum sequence length
        if len(input_ids[0]) > model.config.max_position_embeddings:
            # Truncate the input to fit within the model's limit
            input_ids = input_ids[:, :model.config.max_position_embeddings]

        # Generate the output
        output = model.generate(input_ids)

        if output is not None and len(output) > 0 and output[0] is not None:
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            st.write("### Answer")
            st.write(answer)
        else:
            st.write("### Answer")
            st.write("Sorry, I couldn't generate a valid response.")
except Exception as e:
    st.write("An error occurred:", str(e))
    # You can customize the error message as needed
