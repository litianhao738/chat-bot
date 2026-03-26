import pandas as pd
import json
import time
import re
from atproto import Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# CONFIGURATION
HANDLE = "williamcysit.bsky.social"
APP_PASSWORD = "rjy4-dbcs-hk55-t57b"
QUERY = "entrepreneurship education"
LIMIT = 50
OUTPUT_FILE = "bluesky_sentiment_results.csv"

def clean_social_text(text):
    """
    Cleans text for Project 8 Requirement 3.1: Data Pre-processing.
    Removes URLs, hashtags, and mentions to improve sentiment accuracy.
    """
    text = str(text).lower()
    # Remove URLs (http/https/www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove hashtags and mentions (@user, #topic)
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text, analyzer):
    """
    Performs classification for sentiment analysis (Requirement 169).
    Returns scores and a text-based tag.
    """
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        tag = 'Positive'
    elif compound <= -0.05:
        tag = 'Negative'
    else:
        tag = 'Neutral'
    
    return compound, tag

def run_bluesky_analysis():
    # Initialize API Client and Sentiment Analyzer
    client = Client()
    analyzer = SentimentIntensityAnalyzer()
    
    print(f"--- Logging into Bluesky as {HANDLE} ---")
    try:
        client.login(HANDLE, APP_PASSWORD)
    except Exception as e:
        print(f"Login Failed: {e}")
        return

    print(f"--- Scraping Posts for: '{QUERY}' ---")
    try:
        # Fetch posts using the API
        response = client.app.bsky.feed.search_posts(params={'q': QUERY, 'limit': LIMIT})
        
        raw_data = []
        for post in response.posts:
            # Objective 295: Build Knowledge Base from social media
            raw_data.append({
                "author": post.author.handle,
                "text": post.record.text,
                "created_at": post.record.created_at,
                "reply_count": post.reply_count,
                "like_count": post.like_count
            })
            
        # DATA PROCESSING
        df = pd.DataFrame(raw_data)
        
        df.columns = df.columns.str.strip()
        
        if 'text' not in df.columns:
            print("Error: 'text' column missing from scraped data.")
            return

        print("--- Analyzing Sentiment & Cleaning Data ---")
        df['clean_text'] = df['text'].apply(clean_social_text)
        
        # Apply sentiment analysis
        sentiment_results = df['clean_text'].apply(lambda x: analyze_sentiment(x, analyzer))
        df['sentiment_score'], df['sentiment_tag'] = zip(*sentiment_results)

        # DISPLAY RESULTS
        print("\n--- Project 8 Preliminary Results ---")
        print(f"Total Posts Analyzed: {len(df)}")
        print("\nSentiment Distribution:")
        print(df['sentiment_tag'].value_counts())
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Analyzed data saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Scraping/Analysis Error: {e}")

if __name__ == "__main__":
    run_bluesky_analysis()
