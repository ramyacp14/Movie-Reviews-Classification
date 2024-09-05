'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import string
import sys
import random
import nltk
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter

import sentiment_read_subjectivity
from nltk.collocations import *
from nltk.metrics import ConfusionMatrix
from nltk.classify import DecisionTreeClassifier
import re

# Initialize bigram measures for collocations
bigram_measures = nltk.collocations.BigramAssocMeasures()

# Import lexicons for sentiment analysis
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types('SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')

import sentiment_read_LIWC_pos_neg_words
# initialize positve and negative word prefix lists from LIWC 
#   note there is another function isPresent to test if a word's prefix is in the list
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

# Path to sentiment lexicon
SLpath = "./SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"
SL = sentiment_read_subjectivity.readSubjectivity(SLpath)
SL2 = sentiment_read_subjectivity.read_subjectivity_three_types(SLpath)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# function to extract document features
def document_features(document, word_features):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['V_{}'.format(word)] = (word in document_words)
  return features

# function to extract document features including positive and negative word prefixes
def document_features_list(document, word_features, poslist, neglist):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # Check if any word in the document starts with a positive prefix
    features['contains_positive'] = any(word.startswith(pos) for pos in poslist for word in document)
    # Check if any word in the document starts with a negative prefix
    features['contains_negative'] = any(word.startswith(neg) for neg in neglist for word in document)
    return features

# function to extract features with negation handling
def Not_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
        #go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features

# function to extract bigram features
def bigram_document_features(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)
    return features

# function to extract sentiment lexicon features
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
        features['positivecount'] = weakPos + (5 * strongPos)
        features['negativecount'] = weakNeg + (5 * strongNeg)
    return features

# function to extract features with negation handling for bigrams
def Not_features_bigram(document, word_features, bigram_features, negationwords):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)
    #go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features

# function to extract comprehensive document features including sentiment and negation handling and bigrams
def comprehensive_doc_features(document, word_features, SL, bigram_features, negationwords):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    # NEGATION WORDS: Generate V_ and V_NOT features based on negationwords
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
        features['positivecount'] = weakPos + (5 * strongPos)
        features['negativecount'] = weakNeg + (5 * strongNeg)

    return features

# function to extract comprehensive document features including sentiment, negation, bigrams and neutral word counts
def comprehensive_doc_features_neutral(document, word_features, SL, SL2, bigram_features, negationwords):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    # NEGATION WORDS: Generate V_ and V_NOT features based on negationwords
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
        features['positiveStrengthcount'] = weakPos + (5 * strongPos)
        features['negativeStrengthcount'] = weakNeg + (5 * strongNeg)

    posword = 0
    neutword = 0
    negword = 0
    for word in document_words:
        if word in SL2[0]:
            posword += 1
        if word in SL2[1]:
            neutword += 1
        if word in SL2[2]:
            negword += 1

    features['positivecount'] = posword
    features['neutralcount'] = neutword
    features['negativecount'] = negword

    return features

# function to generate features based on Sentiment and negation
def generateFeatureSets(document, SL, negationwords):
    document_words = set(document)
    # print('LENGTH', len(document_words))
    features = {}
    features['length'] = len(document_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    negationWords = 0
    psc = 0
    nsc = 0
    for word in document_words:
        if word in negationwords:
            negationWords +=1
        features['negationwords'] = negationWords
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            psc = (weakPos) + (strongPos)
            nsc = (weakNeg) + (strongNeg)
        features['positiveStrengthCount'] = (2 * weakPos) + (5 * strongPos)
        features['negativeStrengthCount'] = (2 * weakNeg) + (5 * strongNeg)
    length = len(document_words)
    if length > 10:
        features['percpositive'] = round(psc/length*100,2)
        features['percnegative'] = round(nsc/length*100,2)
    #print(features)
    return features


# function to generate predictions for the test set
def generate_predictions(test_df, classifier, word_features, SL, bigram_features, negationwords):
    predictions = []
    for index, row in test_df.iterrows():
        # Tokenize the phrase
        tokens = nltk.word_tokenize(row['Phrase'])
        # Generate features for the test instance
        test_features = comprehensive_doc_features(tokens, word_features, SL, bigram_features, negationwords)
        # Predict the sentiment label using the classifier
        prediction = classifier.classify(test_features)
        # Append the prediction to the list of predictions
        predictions.append((row['PhraseId'], prediction))
    return predictions

# Function to map predictions to integer labels
def map_predictions(predictions):
    mapped_predictions = []
    for phrase_id, prediction in predictions:
        if prediction == 'negative':
            mapped_predictions.append((phrase_id, 0))
        elif prediction == 'neutral':
            mapped_predictions.append((phrase_id, 2))
        elif prediction == 'positive':
            mapped_predictions.append((phrase_id, 4))
    return mapped_predictions


# function to write predictions to an output file
def write_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        for phrase_id, prediction in predictions:
            f.write("{},{}\n".format(phrase_id, prediction))


# use NLTK to compute evaluation measures from a reflist of gold labels
#    and a testlist of predicted labels for all labels in a list
# returns lists of precision and recall for each label

# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  for phrase in phraselist[:10]:
    print (phrase)
  stop_words = set(stopwords.words('english'))
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  neutraldocs = []
  # add all the phrases
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word.lower() for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if not word.isdigit()]
    tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens]

    #tokens = [word for word in tokens if word.lower() not in stop_words]
    sentiment = int(phrase[1])
    if (sentiment == 2):
        neutraldocs.append((tokens, 'neutral'))
        #phrasedocs.append((tokens, 'neutral'))
    if ((sentiment == 0) or (sentiment == 1)):
        phrasedocs.append((tokens, 'negative'))
    if ((sentiment == 3) or (sentiment == 4)):
        phrasedocs.append((tokens, 'positive'))

  # print a few
  for phrase in phrasedocs[:10]:
    print (phrase)

  docs = []
  for phrase in phrasedocs:
    lowerphrase = ([w.lower() for w in phrase[0]], phrase[1])
    docs.append (lowerphrase)

  # possibly filter tokens
  #stop_words = set(stopwords.words('english'))
  #filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

  # continue as usual to get all words and create word features
  #all_words = nltk.FreqDist(word.lower() for token in filtered_tokens for word in token[0])

  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)
  print(len(all_words))

  #word_features = list(all_words.keys())[:2000]  # Assuming all_words is a dictionary with word frequencies

  finder = BigramCollocationFinder.from_words(all_words_list)
  bigram_features = finder.nbest(bigram_measures.pmi, 250)

  word_items = all_words.most_common(2500)
  word_features = [word for (word, count) in word_items]
  negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather',
                   'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

  # Use different feature functions to generate featuresets

  #doc_featuresets = [(document_features(d, word_features), c) for (d, c) in docs]

  #featuresets = [(document_features_list(d, word_features, poslist, neglist), c) for (d, c) in docs]

  #bigram_featuresets = [(bigram_document_features(d, word_features, bigram_features), c) for (d, c) in docs]

  #SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in docs]

  #featuresets = [(Not_features(d, word_features, negationwords), c) for (d, c) in docs]

  #NOT_featuresets = [(Not_features_bigram(d, word_features, bigram_features, negationwords), c) for (d, c) in docs]

  #new_featuresets = [(generateFeatureSets(d, SL, negationwords), c) for (d, c) in docs]

  #comp_neutral_feature_sets = [(comprehensive_doc_features_neutral(d, word_features, SL, SL2, bigram_features, negationwords), c) for (d, c) in docs]

  comp_feature_sets = [(comprehensive_doc_features(d, word_features, SL, bigram_features, negationwords), c) for (d, c) in docs]

  # train classifier and show performance in cross-validation
  train_set, test_set = comp_feature_sets[round(.3 * int(limit)):], comp_feature_sets[:round(.3 * int(limit))]
  classifier = nltk.NaiveBayesClassifier.train(train_set)

  # Evaluate the classifier on the test set
  test_labels = [label for (_, label) in test_set]
  predicted_labels = [classifier.classify(features) for (features, _) in test_set]

  # Generate the confusion matrix
  cm = ConfusionMatrix(test_labels, predicted_labels)

  # Print the confusion matrix
  print("Confusion Matrix:")
  print(cm)

  print('Overall Accuracy', nltk.classify.accuracy(classifier, test_set))

  # Calculate precision, recall, and F1 score
  TP = cm['positive', 'positive']
  FP = cm['positive', 'negative'] # + cm['positive', 'neutral']
  FN = cm['negative', 'positive'] # + cm['neutral', 'positive']
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f1_score = 2 * (precision * recall) / (precision + recall)

  # Print precision, recall, and F1 score
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1 Score:", f1_score)

  # Read the test data
  test_filepath = os.path.join('test.tsv')
  test_df = pd.read_csv(test_filepath, sep='\t')

  # Generate predictions for the test set
  predictions = generate_predictions(test_df, classifier, word_features, SL, bigram_features, negationwords)
  mapped_predictions = map_predictions(predictions)

  # Write predictions to an output file
  output_filepath = 'predictions.csv'  # Adjust the output filename as needed
  write_predictions(mapped_predictions, output_filepath)

  print("Predictions saved to", output_filepath)




"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""

if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])
