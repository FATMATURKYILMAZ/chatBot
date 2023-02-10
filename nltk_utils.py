import nltk 
from nltk.stem.porter import PorterStemmer
import numpy as np

Stemmer=PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return Stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_word):
     tokenized_sentence=[stem(w) for w in tokenized_sentence]
     bag=np.zeros(len(all_word),dtype=np.float32)
     for idx,w in enumerate(all_word):
         if w in tokenized_sentence:
             bag[idx]=1.0
     return bag        

sentence=["Hello","How","are","you"]
word=["hi","hello","I","you","bye","thank","cool"]
bag=bag_of_words(sentence, word)
print(bag)
# =============================================================================
# a="How lond does shiping day?"
# print(a)
# a=tokenize(a)
# print(a)   
# =============================================================================
words=['organized','organize','organize']
stemmed_word=[stem(w) for w in words]
print(stemmed_word)