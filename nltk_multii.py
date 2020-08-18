import nltk
import re
import string
import random

# Downloading words data
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# Importing required libraries and dependencies
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import request
import json

app = Flask(__name__)

# print(tweet_tokens[0])

# print(pos_tag(tweet_tokens[0]))

with open('price.json') as f:
    priceData = json.load(f)

with open('quality.json') as f:
    qualityData = json.load(f)

with open('taste.json') as f:
    tasteData = json.load(f)

#with open('misc.json') as f:
 #   miscData = json.load(f)

print ("Raw review " , priceData[0])

def noiseRemoval(reviewTokens, stop_words = ()):
   # print("review token", reviewTokens)
    print(stop_words)
    cleaned_tokens = []
   # print("sdfsd")
    for token, tag in pos_tag(reviewTokens):
     #   print("token" , token)
        token = re.sub("[http[s]?://(!@#$;:!*%)(&^~])", '', token)
        token = re.sub(r"http\S+", '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", '', token)
        token = re.sub("[@#:),’]", '', token)
        token = re.sub(r'^https?:\/\/.*[\r\n]*', '', token)
        #token = re.sub("'’", '', token)

    #    print(token," ",tag)
     #   print(token)

        if tag.startswith("VB"): pos = 'v'
        elif tag.startswith('NN'): pos = 'n'
        else: pos = 'a'

        rootWord = WordNetLemmatizer().lemmatize(token, pos)
        rootWord = rootWord.lower()
        if rootWord not in stop_words and rootWord not in string.punctuation:
            cleaned_tokens.append(rootWord)

    return cleaned_tokens

stop_words = stopwords.words('english')

cleanedPriceReviews = []
cleanedQualityReviews = []
cleanedTasteReviews = []
misc_cleaned_tokens_list = []

list = []

cleanedPriceReviews = priceData
#print (cleanedPriceReviews[0])
cleanedQualityReviews = qualityData
cleanedTasteReviews = tasteData

def get_words_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

priceReviewsForModel = get_words_for_model(cleanedPriceReviews)
qualityReviewsForModel = get_words_for_model(cleanedQualityReviews)
tasteReviewsForModel = get_words_for_model(cleanedTasteReviews)

priceDataset = [(tweet_dict, "Price")
                     for tweet_dict in priceReviewsForModel]

qualityDataset = [(tweet_dict, "Quality")
                     for tweet_dict in qualityReviewsForModel]

tasteDataset = [(tweet_dict, "Taste")
                     for tweet_dict in tasteReviewsForModel]

print("Dictionary with Price class : ",priceDataset[1])
print("Dictionary with Quality class : ",qualityDataset[6])
print("Dictionary with Taste class : ",tasteDataset[1])

dataset = priceDataset + qualityDataset + tasteDataset
trainData = dataset[:2000]
testData = dataset[2000:]

Model = NaiveBayesClassifier.train(trainData)

print("Accuracy of the model :", classify.accuracy(Model, testData))
review = "This is in a bad quality."

reviewTokens = noiseRemoval(word_tokenize(review))

# Test print
print(review , " : " , Model.classify(dict([token, True] for token in reviewTokens)))

# Flask API to be used in backend
@app.route("/NlpAspect")
def hello():
    # text as a http get request parameter
    review = request.args.get('text')
    reviewTokens = noiseRemoval(word_tokenize(review))
    return Model.classify(dict([token, True] for token in reviewTokens))

if __name__ == '__main__':
    # Set the API port to 8083 and enable debug to view clear errors
    app.run(debug=True, port=8084)

