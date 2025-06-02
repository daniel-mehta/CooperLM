import torch
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


device_option = st.sidebar.selectbox("Choose Device", ["CPU", "GPU (CUDA)"])
use_gpu = device_option == "GPU (CUDA)" and torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# Load model + tokenizer from huggingface.co
# This is a simple Streamlit app to demonstrate text generation using CooperLM-354M
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("mehta/CooperLM-354M")
    tokenizer = GPT2TokenizerFast.from_pretrained("mehta/CooperLM-354M")
    model.to(device)
    return model, tokenizer

model, tokenizer = load_model()

st.title("🧠 CooperLM-354M Text Generator")
prompt = st.text_area("Enter your prompt:", "Once upon a time,")

temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
max_length = st.slider("Max Length", 10, 200, 100)

if st.button("Generate"):
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.95
    )
    st.markdown("**Generated Text:**")
    st.write(tokenizer.decode(output[0], skip_special_tokens=True))

