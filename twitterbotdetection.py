#import statements to perform sentiment analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

#read data from web-scraping
data = pd.read_csv("twitter_scrape_covid.csv")
tweetlist = data["full_text"].values.tolist()

#iterate through each tweet and calculate the sentiment analysis
for line in tweetlist:
    tweet = line
    print(tweet)
    tweet_words = []

    #split word in case of user mentions and links
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
    
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    #put a space in between found words in tweet
    tweet_proc = " ".join(tweet_words)

    # load model and tokenizer and a pre-trained model
    sentiment = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(sentiment)
    tokenizer = AutoTokenizer.from_pretrained(sentiment)

    #labels for identification
    labels = ['Negative', 'Neutral', 'Positive']

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    #print out predictions for each tweet
    for i in range(len(scores)):
    
        l = labels[i]
        s = scores[i]
        print(l,s)
        print(" ")