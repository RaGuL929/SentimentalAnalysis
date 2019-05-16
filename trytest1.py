import pandas as pd
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,SpatialDropout1D,Dropout
from sklearn.model_selection import train_test_split
import json
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import (
    wordnet,
    stopwords
)

def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

def preprocessing_text(table):
    #put everythin in lowercase
    table['text'] = table['text'].str.lower()
    #Replace rt indicating that was a retweet
    table['text'] = table['text'].str.replace('rt', '')
    #Replace occurences of mentioning @UserNames
    table['text'] = table['text'].replace(r'@\w+', '', regex=True)
    #Replace links contained in the tweet
    table['text'] = table['text'].replace(r'http\S+', '', regex=True)
    table['text'] = table['text'].replace(r'www.[^ ]+', '', regex=True)
    #remove numbers
    table['text'] = table['text'].replace(r'[0-9]+', '', regex=True)
    #replace special characters and puntuation marks
    table['text'] = table['text'].replace(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', regex=True)
    return table
#Replace elongated words by identifying those repeated characters and then remove them and compare the new word with the english lexicon
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

def detect_elongated_words(row):
    regexrep = r'(\w*)(\w+)(\2)(\w*)'
    words = [''.join(i) for i in re.findall(regexrep, row)]
    for word in words:
        if not in_dict(word):
            row = re.sub(word, replace_elongated_word(word), row)
    return row


# Removing stop words in twitter feeds NOTE:- the feed should be in LOWER before passing through this function

def stop_words(table):
    #We need to remove the stop words
    stop_words_list = stopwords.words('english')
    table['text'] = table['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words_list)]))
    return table


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


data = pd.read_csv('Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]


print("*******************Starting data processing*******************")

data = data[data.sentiment != "Neutral"]
data=preprocessing_text(data)
data['text'] = data['text'].apply(lambda x: detect_elongated_words(x))
data['text'] = data['text'].apply(lambda x: handling_negation(x))
data = stop_words(data)
print("*******************Completed data processing********************")



max_features = 3500
tokenizer = Tokenizer(nb_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
# X = tokenizer.texts_to_sequences(data['text'].values)
# X = pad_sequences(X)
Y = pd.get_dummies(data['sentiment']).values
dictionary=tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in data['text'].values:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
X = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print("*******************Building Model*******************")

embed_dim = 128
lstm_out = 196

model = Sequential()
#
# model.add(Embedding(max_features, embed_dim,input_length = X_train.shape[1]))
# model.add(SpatialDropout1D(0.4))
# model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(2, activation='softmax'))
model = Sequential()
model.add(Dense(512, input_shape=(max_features,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
print(model.summary())

print("*******************Training Model*******************")

batch_size = 32
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)
model.save('Sen.h5')  #saving weights of neural network and its architecture in the given file

print("*******************Evalvulating Model*******************")

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
print("*******************Validating Model*******************")

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):

    result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1
print(X_validate[x].reshape(1, X_test.shape[1]).shape)
print("pos_acc", pos_correct / pos_cnt * 100, "%")
print("neg_acc", neg_correct / neg_cnt * 100, "%")