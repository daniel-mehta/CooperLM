# 🧠 CooperLM-354M

**CooperLM** is a 354M parameter GPT-2 based language model trained on a curated subset of English Wikipedia, BookCorpus, and OpenWebText. It was built as a toy project to explore LLM training end-to-end using Hugging Face's Transformers and Datasets libraries.

---

## 📌 Highlights

- 🔢 **Model Size**: 354M parameters
- 🧱 **Architecture**: GPT-2 (24 layers, 16 heads, 1024 hidden size, 256 context length)
- 📚 **Training Data**: 100k samples from cleaned, filtered English text (~688MB)
- ⚙️ **Trained With**: Hugging Face `Trainer`, `fp16`, batch size 16 (gradient accumulation)
- 🧪 **Perplexity**: ~263 on a 1,000-sample evaluation set

---

## 📥 Get the Model

You can:
- 🔗 **Download the trained model** from Hugging Face:
  [https://huggingface.co/mehta/CooperLM-354M](https://huggingface.co/mehta/CooperLM-354M)
- 🛠️ **Train it yourself** using the included notebooks (Requires CUDA Enabled GPU)

> Note: Due to file size constraints, large datasets and model checkpoints are not included in this repo. You’ll need to reprocess data and train locally or modify the training script.

---

## 📂 Project Structure

```
CooperLM/
├── data_processing.ipynb         # Load + clean raw Wikipedia/BookCorpus/OpenWebText
├── tokenize_and_prepare.ipynb    # Tokenize + chunk text into Hugging Face dataset
├── train_model.ipynb             # Train GPT-2 model using Hugging Face Trainer
├── eval_and_generate.ipynb       # Perplexity eval + prompt-based text generation
│
├── raw_data/                     # Cleaned raw text data
│   └── cleaned_data.txt
│
├── tokenized_data/               # Tokenized datasets (HF format)
│   ├── cooper_tokenized_dataset/
│   ├── cooper_subset_200k/
│   └── cooper_subset_100k/
│
├── cooper_model_checkpoints/     # Saved training checkpoints
│   ├── checkpoint-5000/
│   └── checkpoint-5625/
│
├── CooperLM-354M/                # Final exported model + tokenizer
└── README.md
```

---

## 🛠️ How to Use

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained("daniel-mehta/CooperLM-354M")
tokenizer = GPT2TokenizerFast.from_pretrained("daniel-mehta/CooperLM-354M")

prompt = "In a distant future,"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.95, temperature=0.9)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 🧪 Sample Output
Prompt: `The country Malta is`

Generated:
> The country Malta is the largest world of the region. The largest cultural schools are of the central and its city’s world’s city...

## 💻 Training Details
| Setting                | Value           |
| ---------------------- | --------------- |
| Model Type             | GPT2LMHeadModel |
| Epochs                 | 1               |
| Token Count            | \~1.2 million   |
| Context Length         | 256 tokens      |
| Learning Rate          | 5e-4            |
| Precision              | fp16            |
| Batch Size (effective) | 16              |
| GPU                    | RTX 4080        |
| Final Eval Loss        | 5.63            |

---

## ⚠️ Disclaimer
This is a toy model for learning and experimentation. Outputs may be inaccurate or nonsensical. Do not use in production without further tuning.

---

## 🚀 What I Learned

- How to build and train a GPT-style transformer from scratch
- Practical experience with tokenizer prep, entropy filtering, and data chunking
- Fine-tuning loops and managing memory on a consumer GPU

---
## 🔭 Next Steps

- Enable training or inference on CPU
- Add a Streamlit or Gradio demo

---

## 📝License
MIT
