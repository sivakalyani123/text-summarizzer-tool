# 📝 Text Summarization Tool

This is a Python-based text summarization tool that uses Natural Language Processing (NLP) techniques to generate concise summaries of lengthy input texts. It uses a pre-trained model (`facebook/bart-large-cnn`) from Hugging Face Transformers.

## 🚀 Features

- Accepts custom user input from the command line
- Automatically summarizes long articles or paragraphs
- Uses state-of-the-art NLP models (Abstractive Summarization)
- Easy to run and modify

---

## 📂 Project Structure

text-summarization-tool/
│
├── text_summarizer_input.py # Main Python script
├── README.md # Project documentation
└── requirements.txt # Required Python packages

🛠️ Requirements

- Python 3.8+
- Transformers
- Torch

📦 Installation

1. Clone this repository or download the files.
2. Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt

Or install packages manually:
pip install transformers torch

▶️ How to Run
python text_summarizer_input.py
