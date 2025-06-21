# 🐶 CooperLM-354M

**CooperLM** is a 354M-parameter language model based on the GPT-2 architecture, trained from scratch on a curated subset of English Wikipedia, BookCorpus, and OpenWebText. It was built as a toy project to explore end-to-end LLM training using Hugging Face’s Transformers and Datasets libraries. 

*In memory of Cooper — my puppy and companion for 14 years.*

---

## 📌 Highlights

- 🔢 **Model Size**: 354M parameters  
- 🧱 **Architecture**: GPT-2 (24 layers, 16 heads, 1024 hidden size, 256 context length)  
- 📚 **Training Data**: 100k samples from cleaned, filtered English text (~688MB)  
- ⚙️ **Trained With**: Hugging Face `Trainer`, `fp16`, batch size 16 (gradient accumulation)  
- 🧪 **Perplexity**: ~263 on a 1,000-sample evaluation set (expectedly high due to limited hardware)  

---

## 📥 Get the Model

You can:
- 🔗 **Download the trained model** from Hugging Face:  
  [https://huggingface.co/mehta/CooperLM-354M](https://huggingface.co/mehta/CooperLM-354M)

- 📦 **Try the quantized version** (4-bit, GGUF-ready):  
  [https://huggingface.co/mehta/CooperLM-354M-4bit](https://huggingface.co/mehta/CooperLM-354M-4bit)

- 🛠️ **Train it yourself** using the included notebooks (Requires CUDA-enabled GPU)

---

## 📂 Project Structure

```
CooperLM/
├── data_processing.ipynb           # Load + clean raw Wikipedia/BookCorpus/OpenWebText
├── tokenize_and_prepare.ipynb      # Tokenize + chunk text into Hugging Face dataset
├── train_model.ipynb               # Train GPT-2 model using Hugging Face Trainer
├── eval_and_generate.ipynb         # Perplexity eval + prompt-based text generation
│
├── streamlit_app/                  # Streamlit UI for interactive prompt generation
│   └── app.py
│
├── CooperLM-354M/                  # Final exported model + tokenizer
├── CooperLM-354M-quantized/        # Folder for quantized model (4-bit)
├── quantize_model.ipynb            # Notebook for quantizing the trained model
├── requirements.txt                # Dependencies for training + Streamlit UI
└── README.md
```

---

## 🛠️ How to Use (Full Model)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mehta/CooperLM-354M")
tokenizer = AutoTokenizer.from_pretrained("mehta/CooperLM-354M")

prompt = "In a distant future,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🧪 Example Output (Toy Model):
Prompt: `The country Malta is`

Generated:
> The country Malta is the largest world of the region. The largest cultural schools are of the central and its city’s world’s city...

---

## 🧠 Quantization Notebook

To reduce model size and enable lightweight inference, the `quantize_model.ipynb` file demonstrates how to:

- Load any compatible Hugging Face model  
- Apply 4-bit quantization using `AutoGPTQForCausalLM`  
- Save and upload to a Hugging Face repo  
- Run basic generation tests with the quantized model

---

## 🌐 Streamlit Demo

A lightweight Streamlit app is included to run text generation in-browser using CooperLM-354M.

```bash
cd streamlit_app
streamlit run app.py
```

It auto-detects whether to use CPU or GPU and allows device selection via sidebar.

---

## 🚀 What I Learned

- How to build and train a GPT-style transformer from scratch  
- Practical experience with tokenizer prep, entropy filtering, and data chunking  
- Fine-tuning loops and managing memory on a consumer GPU  
- How quantization makes small models practical and shareable

---

## 📝 License

This project is licensed under the MIT License. Feel free to use or build on it for your own experiments.
