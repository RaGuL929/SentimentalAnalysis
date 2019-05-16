import keras
import pickle
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import (
    wordnet,
    stopwords
)
from keras import backend as K
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
import keras.preprocessing.text as kpt
def convert_text_to_index_array(text):
    with open('dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        # else:
            # print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

def load_model1(filename):

    model= keras.models.load_model(filename)
    # model.summary()
    return model

def preprocessing_text(text):
    #put everythin in lowercase
    text=text.lower()
    #Replace rt indicating that was a retweet
    text=text.replace('rt', '')
    #Replace occurences of mentioning @UserNames
    text = re.sub(r'@\w+', "", text)

    # text= text.replace(r'@\w+', '', regex=True)

    #Replace links contained in the tweet
    text = re.sub(r'http\S+', "", text)

    # text = text.replace(r'http\S+', '', regex=True)
    text = re.sub(r'www.[^ ]+', "", text)

    #remove numbers
    text = re.sub(r'[0-9]+', "", text)

    #replace special characters and puntuation marks
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', "", text)

    return text

def in_dict(word):
    if wordnet.synsets(word):
        #if the word is in the dictionary, it will return True
        return True

def replace_elongated_word(word):
    regex = r'(\w*)(\w+)\2(\w*)'
    repl = r'\1\2\3'
    if in_dict(word):
        return word
    new_word = re.sub(regex, repl, word)
    if new_word != word:
        return replace_elongated_word(new_word)
    else:
        return new_word

def detect_elongated_words(text):
    regexrep = r'(\w*)(\w+)(\2)(\w*)'
    words = [''.join(i) for i in re.findall(regexrep, text)]
    for word in words:
        if not in_dict(word):
            text = re.sub(word, replace_elongated_word(word), text)
    return text


def stop_words(text):
    #We need to remove the stop words
    stop_words_list = stopwords.words('english')

    text=' '.join([word for word in text.split() if word not in (stop_words_list)])
    return text


#replacing negations with antonyms
def replace_antonyms(word):
    # We get all the lemma for the word
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # if the lemma is an antonyms of the word
            if lemma.antonyms():
                # we return the antonym
                return lemma.antonyms()[0].name()
    return word



def handling_negation(row):
    # Tokenize the row
    words = word_tokenize(row)
    speach_tags = ['JJ', 'JJR', 'JJS', 'NN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP']
    tags = nltk.pos_tag(words)
    tags_2 = ''
    if "n't" in words and "not" in words:
        tags_2 = tags[min(words.index("n't"), words.index("not")):]
        words_2 = words[min(words.index("n't"), words.index("not")):]
        words = words[:(min(words.index("n't"), words.index("not"))) + 1]
    elif "n't" in words:
        tags_2 = tags[words.index("n't"):]
        words_2 = words[words.index("n't"):]
        words = words[:words.index("n't") + 1]
    elif "not" in words:
        tags_2 = tags[words.index("not"):]
        words_2 = words[words.index("not"):]
        words = words[:words.index("not") + 1]

    for index, word_tag in enumerate(tags_2):
        if word_tag[1] in speach_tags:
            words = words + [replace_antonyms(word_tag[0])] + words_2[index + 2:]
            break

    return ' '.join(words)



def preprocess_text(text):
        text=preprocessing_text(text)
        text=detect_elongated_words(text)
        text=handling_negation(text)
        text=stop_words(text)

        return text
def predict_text(text):
    K.clear_session() #clear session is mandatory if u want to run the program as service
    text=preprocess_text(text)
    filename = 'Sen.h5'
    model=load_model1(filename)

    labels = ['negative', 'positive']


    tokenizer = Tokenizer(num_words=3500)

    testArr = convert_text_to_index_array(text)
    input = tokenizer.sequences_to_matrix([testArr], mode='binary')
    # predict which bucket your input belongs in
    pred = model.predict(input)
    K.clear_session()#clear session is mandatory if u want to run the program as service
    # returning label and its confidence
    return (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100)
