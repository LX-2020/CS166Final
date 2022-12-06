# @author: Lexin Ma
# 11/26/2022
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from wordcloud import WordCloud
import csv
import matplotlib.pyplot as plt

# read input file
file = "twitterData.csv"
columns = []
rows = []

with open(file, 'r') as csvFile:
    csvreader = csv.reader(csvFile)
    columns = next(csvreader)

    for row in csvreader:
        rows.append(row)

# import model
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

# analyze tweets
res = []
for tweet in rows:
    tweet_words = []
    scores_dict = {}
    # import pdb;pdb.set_trace()
    for word in tweet[1].split(" "):
        if word.startswith('@') and len(word) > 1:
            continue
        elif word.startswith('http'):
            continue
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    # print(tweet_proc)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # print(encoded_tweet)
    output = model(**encoded_tweet)
    # print(output)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {'negative': scores[0], 'neutral': scores[1], 'positive': scores[2]}
    # print(scores_dict)
    find_max = max(scores_dict, key=scores_dict.get)
    # print(find_max)
    res.append((tweet_proc, find_max))

res_emo = [x[1] for x in res]
res_count = {'negative': res_emo.count('negative'), 'neutral': res_emo.count('neutral'),
             'positive': res_emo.count('positive')}
print(res_count)

words_pos = []
words_neg = []
words_all = []
# import pdb;pdb.set_trace()
for t, emo in res:
    words_all.extend(t.split(" "))
    if emo == 'negative':
        words_neg.extend(t.split(" "))
    elif emo == 'positive':
        words_pos.extend(t.split(" "))


def count_unique(l):
    chars_to_remove = ['.', ',', '!', '#', '\\', '/', '"', '(', ')', '😳', '🥳', '\n', '?', '“']
    out = dict()
    for x in l:
        x = x.lower()
        if any([c in x for c in chars_to_remove]):
            continue
        if x not in out:
            out[x] = 0
        out[x] += 1
    return out


# import pdb;pdb.set_trace()
dict_pos = count_unique(words_pos)
dict_neg = count_unique(words_neg)
dict_all = count_unique(words_all)

dict_pos_ratio = dict((k, 1.0 * dict_pos[k] / dict_all[k]) for k in dict_pos)
dict_neg_ratio = dict((k, 1.0 * dict_neg[k] / dict_all[k]) for k in dict_neg)

sorted_dict_pos_ratio = dict(sorted(dict_pos_ratio.items(), key=lambda item: item[1], reverse=True))
sorted_dict_neg_ratio = dict(sorted(dict_neg_ratio.items(), key=lambda item: item[1], reverse=True))

print(list(sorted_dict_pos_ratio.keys())[:50])
print(list(sorted_dict_neg_ratio.keys())[:50])

#draw pie chart
plt.title('Sentiment Analysis')
plt.pie(res_count.values(), labels=res_count.keys())
plt.show()

# draw word cloud
unique_string = " ".join(list(sorted_dict_neg_ratio.keys())[:50])
wordcloud = WordCloud(width=1000, height=500).generate(unique_string)
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("your_file_name" + ".png", bbox_inches='tight')
plt.show()
