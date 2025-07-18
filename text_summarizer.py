# text_summarizer.py

from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text (can be long article)
input_text = """
India is one of the world’s fastest-growing economies and a major player in global trade. Over the past few decades,
the country has transitioned from an agriculture-based economy to a service-dominated one, with information technology
and business outsourcing services leading the charge. However, the manufacturing sector is also being boosted through
initiatives like Make in India, aimed at increasing domestic production and reducing import dependency.
Furthermore, India has a rapidly growing middle class and is seeing major investments in infrastructure,
education, and healthcare. These advancements are positioning the country to be a leader in the 21st-century global economy.
"""

# Generate summary
summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)

# Display results
print("\n--- Original Text ---")
print(input_text)
print("\n--- Summarized Text ---")
print(summary[0]['summary_text'])
