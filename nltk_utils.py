import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer

#nltk.download('punkt')

stemmer = PorterStemmer()


def tokenizer(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_wrds(tokenized_sentence, all_words):

    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1
    return bag


