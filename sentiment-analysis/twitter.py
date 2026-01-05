import tweepy
 
# Add Twitter API key and secret
consumer_key = "***"
consumer_secret = "****"

#auth = tweepy.OAuth2AppHandler(consumer_key, consumer_secret)
#auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

# Authenticate with the API v1.1
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_token_secret)
#api_v1 = tweepy.API(auth)


# Create a wrapper for the Twitter API
#api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

api = tweepy.API(auth)

tweets = api.search_tweets(q="adobe Bridge",count=3)
print(tweets)
