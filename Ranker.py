import sqlite3
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the Pegasus model and tokenizer
model_name = "google/pegasus-large"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Connect to your SQLite database
connection = sqlite3.connect('news_articles.db')
cursor = connection.cursor()

# Check if the 'Rank' column exists, and if not, add it
cursor.execute("PRAGMA table_info(news_articles)")
columns = cursor.fetchall()
if ('Rank', 'INTEGER', 0) not in columns:
    cursor.execute("ALTER TABLE news_articles ADD COLUMN Rank INTEGER")

# Check if the 'Ranking_Score' column exists, and if not, add it
if ('Ranking_Score', 'REAL', 0) not in columns:
    cursor.execute("ALTER TABLE news_articles ADD COLUMN Ranking_Score REAL")

# Retrieve the articles for ranking
cursor.execute("SELECT id, article_content FROM news_articles")
articles = cursor.fetchall()

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Create a list to store (article_id, sentiment_score) tuples
article_scores = []

# Iterate through retrieved articles, summarize, and calculate ranking scores
for article in articles:
    article_id, article_content = article
    input_ids = tokenizer.encode(article_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Calculate sentiment score
    sentiment_score = sid.polarity_scores(summary)['compound']
    
    # Append (article_id, sentiment_score) tuple to the list
    article_scores.append((article_id, sentiment_score))

# Sort the article scores based on the magnitude of the sentiment score (absolute value)
sorted_article_scores = sorted(article_scores, key=lambda x: abs(x[1]), reverse=True)

# Update the ranking score in the database based on the sorted scores
for idx, (article_id, sentiment_score) in enumerate(sorted_article_scores, start=1):
    cursor.execute("UPDATE news_articles SET Ranking_Score = ? WHERE id = ?", (sentiment_score, article_id))
    
    # Update the 'Rank' column with the rank of each article
    cursor.execute("UPDATE news_articles SET Rank = ? WHERE id = ?", (idx, article_id))

# Commit changes and close the database connection
connection.commit()
connection.close()