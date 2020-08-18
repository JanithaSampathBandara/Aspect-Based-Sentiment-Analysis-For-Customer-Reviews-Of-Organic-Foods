import nltk
import re
import string
import random

#Downloading words data
#nltk.download('twitter_samples')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')

# Importing required libraries and dependencies
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import request

app = Flask(__name__)

def noiseRemoval(reviewTokens, stop_words = ()):
  #  print("review token", reviewTokens)
    cleaned_tokens = []

    for token, tag in pos_tag(reviewTokens):
        token = re.sub("[http[s]?://(!@#$;:!*%)(&^~])", '', token)
    #    print("token" , token)
        token = re.sub(r"http\S+", '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", '', token)
        token = re.sub("[@#:),’]", '', token)
        token = re.sub(r'^https?:\/\/.*[\r\n]*', '', token)
        #token = re.sub("'’", '', token)
        #print(token," ",tag)
        #print(token)

        if tag.startswith("VB"): pos = 'v'
        elif tag.startswith('NN'): pos = 'n'
        else: pos = 'a'

        rootWord = WordNetLemmatizer().lemmatize(token, pos)
        rootWord = rootWord.lower()
        if rootWord not in stop_words and rootWord not in string.punctuation:
            cleaned_tokens.append(rootWord)

    return cleaned_tokens

positiveReviewTokens = twitter_samples.tokenized('positive_tweets.json')
negativeReviewTokens = twitter_samples.tokenized('negative_tweets.json')

print("pos before cleaned",positiveReviewTokens[0])

cleanedPositiveReviewTokens = []
cleanedNegativeReviewTokens = []

# Remove noise with above function noiseRemoval()
for words in positiveReviewTokens:
  #  print(words) a review from all review list
    cleanedPositiveReviewTokens.append(noiseRemoval(words, stopwords.words('english')))

for words in negativeReviewTokens:
    cleanedNegativeReviewTokens.append(noiseRemoval(words, stopwords.words('english')))

print("pos after cleaned and 2d",cleanedPositiveReviewTokens[0])
print("cleaned and appended " , cleanedPositiveReviewTokens)

def createDictionary(cleanedReviews):
    #taking one review at a time from list of reviews
    for reviewTokens in cleanedReviews:
        yield dict([token, True] for token in reviewTokens)

positiveReviewsForModel = createDictionary(cleanedPositiveReviewTokens)
negativeReviewsForModel = createDictionary(cleanedNegativeReviewTokens)
#print(positiveReviewsForModel[0])

positiveReviewDataset = [(tweet_dict, "Positive")
                     for tweet_dict in positiveReviewsForModel]

negativeReviewDataset = [(tweet_dict, "Negative")
                     for tweet_dict in negativeReviewsForModel]

print("Dictionary with Positive class : ",positiveReviewDataset[7])
print("Dictionary with Negative class : ",negativeReviewDataset[7])
#print("tagged neg :",negative_dataset[0]) 

dataset = positiveReviewDataset + negativeReviewDataset

print("Dataset[0] :",dataset[0])
print("Dataset length",len(dataset))

random.shuffle(dataset)

trainData = dataset[:7000]
testData = dataset[7000:]

trainedModel = NaiveBayesClassifier.train(trainData)

print("Accuracy of the model : ", classify.accuracy(trainedModel, testData))

review = "This is a bad product."
reviewTokens = noiseRemoval(word_tokenize(review))

# Test print
print(review , " : " , trainedModel.classify(dict([token, True] for token in reviewTokens)))

#Text = "j@nittha"
#Text = re.sub("@", "a", Text)
#print(Text)

# Flask API to be used in backend
@app.route("/NLP")
def hello():
    # text as a http get request parameter
    consumerReview = request.args.get('text')
    consumerReviewTokens = noiseRemoval(word_tokenize(consumerReview))
    return trainedModel.classify(dict([token, True] for token in consumerReviewTokens))

if __name__ == '__main__':
    # Set the API port to 8083 and enable debug to view clear errors
    app.run(debug=True, port=8083)

