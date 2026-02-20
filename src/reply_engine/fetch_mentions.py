import os
import tweepy
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
if not bearer_token:
    print("Missing TWITTER_BEARER_TOKEN in .env")
    exit(1)

client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

query = '@OpenClawAI -from:OpenClawAI lang:en -is:retweet'
tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=['author_id', 'public_metrics', 'created_at', 'conversation_id', 'context_annotations', 'referenced_tweets'])

if not tweets.data:
    print("No recent mentions found.")
else:
    print(f"Recent @OpenClawAI mentions ({len(tweets.data)}):")
    for i, tweet in enumerate(tweets.data, 1):
        author = tweet.author_id
        try:
            user = client.get_user(id=author, user_fields=['username', 'name']).data
            username = user.username
            name = user.name
        except:
            username = name = 'Unknown'
        print(f"{i}. @{username} ({name}): {tweet.text}")
        print(f"   ID: {tweet.id} | Likes: {tweet.public_metrics['like_count']} | Time: {tweet.created_at}")
        print(f"   URL: https://twitter.com/{username}/status/{tweet.id}\n")