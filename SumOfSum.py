import sqlite3
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import datetime

# Load the Pegasus model and tokenizer
model_name = "google/pegasus-large"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Connect to your SQLite database
connection = sqlite3.connect('news_articles.db')
cursor = connection.cursor()

# Define the time span (e.g., last one hour or last one day)
current_time = datetime.datetime.now()
time_span = datetime.timedelta(hours=1)  # You can change this to 'days' for one day

# Calculate the start time based on the time span
start_time = current_time - time_span

# Query the database to retrieve the summaries of news articles within the time span
cursor.execute("SELECT summary FROM news_articles WHERE published_date >= ? AND published_date <= ?", (start_time, current_time))
summaries = cursor.fetchall()

# Close the database connection
connection.close()

# Concatenate the retrieved summaries to create a summary of summaries
summary_of_summaries = "\n".join([summary[0] for summary in summaries])

# Use the Pegasus model to generate a concise summary of the concatenated summaries
input_ids = tokenizer.encode(summary_of_summaries, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print or use the 'final_summary' as needed
print("Summary of Summaries:")
print(final_summary)
