import sqlite3
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from gnews import GNews
import requests
from bs4 import BeautifulSoup

# Load the Pegasus model and tokenizer
model_name = "google/pegasus-large"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

def fetch_news_by_keyword(keyword, start_date, end_date, max_results):
    google_news = GNews()
    google_news.start_date = start_date
    google_news.end_date = end_date
    google_news.max_results = max_results

    result = google_news.get_news(keyword)
    return result

def extract_article_content(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the article content (you may need to adapt this based on the website structure)
        article_content = ""
        paragraphs = soup.find_all('p')  # Assuming article text is wrapped in <p> tags
        for paragraph in paragraphs:
            article_content += paragraph.get_text() + "\n"
        
        return article_content
    else:
        print(f"Failed to fetch article from {url}")
        return None

if __name__ == "__main__":
    keyword = '"WORLD"'
    start_date = (2022, 1, 1)
    end_date = (2022, 2, 1)
    max_results = 3

    news_articles = fetch_news_by_keyword(keyword, start_date, end_date, max_results)

    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('news_articles.db')
    cursor = conn.cursor()

    # Create a table to store news articles (if it doesn't exist)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            published_date TEXT,
            article_content TEXT,
            summary TEXT,
            url TEXT,
            publisher TEXT
        )
    ''')

    # Insert data into the database and generate summaries
    for article in news_articles:
        title = article['title']
        description = article['description']
        published_date = article['published date']
        article_content = extract_article_content(article['url'])  # You can use your extraction logic here
        url = article['url']
        publisher = article['publisher']['title']
        
        # Insert the data into the database
        cursor.execute('''
            INSERT INTO news_articles (title, description, published_date, article_content, url, publisher)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, description, published_date, article_content, url, publisher))
        
        # Tokenize the article content and generate the summary
        input_ids = tokenizer.encode(article_content, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Update the "summary" column with the generated summary
        cursor.execute("UPDATE news_articles SET summary = ? WHERE url = ?", (summary, url))

    # Commit changes and close the database connection
    conn.commit()
    conn.close()
