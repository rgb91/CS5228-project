import numpy as np
import pickle
from features_classwise import read_keywords, read_labels


ids, keywords = zip (*read_keywords ().items ())
categories = []
cat_dict = read_labels ()
for id in ids:
    categories.append (cat_dict[id])

# Load all the pre-saved tf-idf models
tfidf_vect_ngram = []
for label in range(0, 5):
    tfidf_vect_ngram[label] = pickle.load(open("tfidf_vect_ngram"+str(label)+".pickle"), "rb")

for keywordlist, label in zip (keywords, categories):
    tfidf_vect_ngram[label].transform()