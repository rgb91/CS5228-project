from nltk import WordNetLemmatizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import numpy as np
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from collections import Counter
import re
import pickle
from nltk.corpus import wordnet  # To get words in dictionary with their parts of speech


def get_pos(word):
    w_synsets = wordnet.synsets (word)
    pos_counts = Counter ()
    pos_counts["n"] = len ([item for item in w_synsets if item.pos () == "n"])
    pos_counts["v"] = len ([item for item in w_synsets if item.pos () == "v"])
    pos_counts["a"] = len ([item for item in w_synsets if item.pos () == "a"])
    pos_counts["r"] = len ([item for item in w_synsets if item.pos () == "r"])

    most_common_pos_list = pos_counts.most_common (3)
    return most_common_pos_list[0][
        0]  # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )


def lemmatize(tokens_list):
    """
        Uses WordNet lemmatizer to lemmatize
    """
    wordnet_lemmatizer = WordNetLemmatizer ()
    lemmatized_list = []
    for i in tokens_list:
        lemmatized_list.append (wordnet_lemmatizer.lemmatize (i, get_pos (i)))
    return lemmatized_list


def read_titles():
    """
        Function read the titles file and stores it in a list in the format (id, title)
        where id is the docid
    """
    titles = {}
    with open ("newdata/train_v_title", "r") as fp:
        for line in fp:
            id, sent = line.lower ().split ("\t")
            titles.update ({
                id: sent.strip ()
            })
    print ("Length of titles: ", len (titles))
    return titles


def read_keywords():
    """
        Function reads the news articles' keywords files : Thread_[0..5]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = 6
    titles = read_titles ()
    lines = []
    for num_file in range (num_files):
        file_path = "newdata/Thread_keywords_" + str (num_file) + ".dat"
        with open (file_path, "r") as fp:
            lines += fp.read ().split ("###")
    lines = map (lambda line: line.split ("||"), lines)
    keywords = {}
    empty = 0
    ids_set = set ()
    for line in lines:
        if len (line) >= 2:
            id, others = line[0], ' '.join (line[1:]).strip ()
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + re.sub (",", " ", others)
            keywords.update ({
                id: preprocess (others.lower ())
            })
            ids_set.add (id)
    print ("Missing ids: ", set (titles.keys ()).difference (ids_set))

    # filter all those lines that have Empty as there article text
    return keywords


def read_news_articles():
    """
        Function reads the news articles files : Thread_[0..3]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = 6
    titles = read_titles ()
    lines = []
    for num_file in range (num_files):
        file_path = "Thread_" + str (num_file) + ".dat"
        with open (file_path, "r") as fp:
            lines += fp.read ().split ("###")
    lines = map (lambda line: line.split ("||"), lines)
    articles = {}
    empty = 0
    ids_set = set ()
    for line in lines:
        if len (line) >= 2:
            id, others = line[0], ' '.join (line[1:]).strip ()
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + others
            articles.update ({
                id: preprocess (others.lower ())
            })
            ids_set.add (id)
    print ("Missing ids: ", set (titles.keys ()).difference (ids_set))

    # filter all those lines that have Empty as there article text
    return articles


def read_labels():
    category = {}
    with open ("newdata/train_v_category", "r") as fp:
        for line in fp:
            id, sent = line.lower ().split ("\t")
            category.update ({
                id: int (sent.strip ())
            })
    return category


def preprocess(article):
    # Remove non-alphanumeric characters
    processed_line = re.sub (r'\W+', ' ', article).strip ()
    # Remove all digits
    processed_line = re.sub (r'\w*\d\w*', '', processed_line).strip ()
    # Remove non-ascii characters
    processed_line = processed_line.encode ("ascii", errors="ignore").decode ()
    # Lemmatize
    processed_line = lemmatize (processed_line.split (" "))
    return processed_line


def print_top_n_features(vectorizer, n):
    indices = np.argsort (vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names ()
    top_features = [features[i] for i in indices[:n]]
    print (top_features)


ids, keywords = zip (*read_keywords ().items ())
cat_dict = read_labels ()
categories = []

for id in ids:
    categories.append (cat_dict[id])

new_keywords = []
new_categories = []
classwise_keywords = dict ({0: '', 1: '', 2: '', 3: '', 4: ''})

for keywordlist, label in zip (keywords, categories):
    # if label != 4:
    if True:
        new_keywords.append (keywordlist)
        new_categories.append (label)
        classwise_keywords.update ({
            label: classwise_keywords[label] + ' ' + str (keywordlist)
        })
        # classwise_keywords[label].append (keywordlist)

keywords = new_keywords
categories = new_categories

for label, keywords_for_one_label in classwise_keywords.items ():
    print ()
    print ('Label: ' + str (label))
    # CountVectorizer Models
    # count_vect = CountVectorizer (analyzer='word', token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1, 2), min_df=0.004, max_df=50)
    # count_vect.fit (keywords_for_one_label)
    # print (count_vect.get_feature_names ())
    # x_count = count_vect.transform (keywords_for_one_label)

    # word level tf-idf
    # tfidf_vect = TfidfVectorizer (analyzer='word', token_pattern=r'\w{1,}', stop_words='english', max_features=5000)
    # tfidf_vect.fit ([keywords_for_one_label])
    # x_tfidf = tfidf_vect.transform ([keywords_for_one_label])
    # tfidf_vect.fit (keywords_for_one_label)
    # x_tfidf = tfidf_vect.transform (keywords_for_one_label)
    # print_top_n_features (tfidf_vect, 100)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer (analyzer='word', token_pattern=r'\w{1,}', stop_words='english',
                                        ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit ([keywords_for_one_label])
    x_tfidf_ngram = tfidf_vect_ngram.transform ([keywords_for_one_label])
    # tfidf_vect_ngram.fit (keywords_for_one_label)
    # x_tfidf_ngram = tfidf_vect_ngram.transform (keywords_for_one_label)
    print_top_n_features (tfidf_vect_ngram, 10)
    # pickle.dump(tfidf_vect_ngram, open("save/tfidf_vect_ngram_"+str(label)+".pickle", "wb"))
    # pickle.dump(x_tfidf_ngram, open("save/x_tfidf_ngram_"+str(label)+".pickle", "wb"))
