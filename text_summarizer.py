# text_summarizer_input.py

from transformers import pipeline

# Load summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Get user input
print("Enter the text you want to summarize (press Enter twice to submit):\n")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

input_text = " ".join(lines)

# Validate length
if len(input_text.split()) < 30:
    print("\nPlease enter a longer text (minimum ~30 words for meaningful summarization).")
else:
    # Generate summary
    summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)

    # Show results
    print("\n--- Original Text ---")
    print(input_text)
    print("\n--- Summarized Text ---")
    print(summary[0]['summary_text'])
