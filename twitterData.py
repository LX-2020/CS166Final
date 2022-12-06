import snscrape.modules.twitter as sntwitter
import pandas as pd

#query = "covid vaccine until:2022-11-10 since:2022-10-01"
query = "US election until:2022-11-23 since:2022-10-01"
tweets = []
limit = 200

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.content])
df = pd.DataFrame(tweets, columns=['Tweets'])
print(df)
df.to_csv('twitterData.csv')
