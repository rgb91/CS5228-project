"""
    Vocabulary builder from titles column.
"""
import re
import numpy as np
import pickle
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import wordnet # To get words in dictionary with their parts of speech


def get_pos(word):
        w_synsets = wordnet.synsets(word)
        pos_counts = Counter()
        pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
        pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
        pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
        pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )

        most_common_pos_list = pos_counts.most_common(3)
        return most_common_pos_list[0][0] # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )

def read_titles():
    """
        Function read the titles file and stores it in a list in the format (id, title)
        where id is the docid
    """
    titles = {}
    with open("titles", "r") as fp:
        for line in fp:
            try:
                id, sent = line.lower().split("|")
            except:
                splitted = line.lower().split("|")
                id,sent = splitted[0], ' '.join(splitted[1:])

            titles.update({
                id: sent.strip()
            })
    return titles

def read_news_articles():
    """
        Function reads the news articles files : Thread_[0..3]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = 4
    titles =  read_titles()
    lines = []
    for num_file in range(num_files):
        file_path = "test_data_v2/Thread_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            lines += fp.read().split("##")
    lines = map(lambda line: line.split("|"), lines)
    articles  = {}
    empty = 0
    for line in lines:
        if len(line) >= 2:
            id, others = line[0], ' '.join(line[1:]).strip()
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + others
            articles.update({
                id: others.lower().split("\n")
            })
    return articles

def preprocess(lines):
    """
        Does some preprocessing.
    """
    processed_lines = []
    for line in lines:
        #Remove non-alphanumeric characters
        processed_line = re.sub(r'\W+', ' ', line).strip()
        #Remove all digits
        processed_line = re.sub(r'\w*\d\w*', '', processed_line).strip()
        if processed_line:
            processed_lines.append(processed_line)
    return processed_lines

def tokenize(lines):
    """
        Uses nltk word_tokenize to tokenize the lines
    """
    tokenized_lines = []
    for line in lines:
        tokenized_lines.append(word_tokenize(line))
    return tokenized_lines

def remove_stopwords(tokenized_lines):
    """
        Remove all the stopwords
    """
    lines = []
    stop_words = stopwords.words("english")
    for line in tokenized_lines:
        lines.append([word for word in line if word not in stop_words])
    return lines

def stem(tokens_list):
    """
        Uses Porter Stemmmingalgorithm to stem the lines.
    """
    stemmed_list = []
    p_stemmer = PorterStemmer()
    for token_list in tokens_list:
        stemmed_list.append([p_stemmer.stem(i) for i in token_list])
    return stemmed_list

def lemmatize(tokens_list):
    """
        Uses WordNet lemmatizer to lemmatize
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    for token_list in tokens_list:
        lemmatized_list.append([wordnet_lemmatizer.lemmatize(i, get_pos(i)) for i in token_list])
    return lemmatized_list

def tag_docs(tokens_list):
    """
        Creates a list of Tagged documents
    """
    tagged_docs = []
    for index, token_list in enumerate(tokens_list):
        tagged_docs.append(TaggedDocument(token_list, str(index)))
    return tagged_docs

def get_representation(preliminary_count, size):
    """
        It gets the bag-of-words representation of a word.
    """
    representation = []
    for idx, count in preliminary_count:
        if len(representation) == idx:
            representation.append(count)
        else:
            while len(representation) < idx:
                representation.append(0)
            representation.append(count)

    while len(representation) < size:
        representation.append(0)
    return representation

def create_bow(stemmed_tokens):
    """
        Creates the bow representation for tokens
    """
    texts = []
    for lines in stemmed_tokens:
        texts += lines
    dictionary = corpora.Dictionary(texts)
    dictionary.save('dictionary.dict')

    size_of_dictionary = len(dictionary)
    #Get the doc 2 bag-of-words model
    bow_representation = []
    id_list = []
    for lines in stemmed_tokens:
        para = []
        for line in lines:
            para += line
        bow_representation.append(get_representation(dictionary.doc2bow(para), size_of_dictionary))
    return np.array(bow_representation)

def complete_preprocessing(lines):
    """
        Does a series of preprocessing steps.
    """
    print("Precessing")
    lines = map(lambda line: preprocess(line), lines)
    print("Tokenizing")
    lines = map(lambda line: tokenize(line), lines)
    print("Removing Stop words")
    lines = map(lambda line: remove_stopwords(line), lines)
    #lines = map(lambda tup: (tup[0], stem(tup[1])), stopword_removed_lines)
    print("Lemmatizing")
    lines = map(lambda line: lemmatize(line), lines)
    return lines

def augment_with_title(lines, all_lines):
    """
        For all those documents that had bad urls, use the title instead
    """
    out = []
    curr_id = 0
    for id, text in lines:
        if id == str(curr_id):
            out.append((id, text))
        else:
            while str(curr_id) != id:
                out.append(all_lines[curr_id])
                curr_id += 1
            out.append((id, text))
        curr_id += 1
    return out
"""
print("Reading articles")
articles_map = read_news_articles()
ids, articles = [], []
for id, article in articles_map.iteritems():
    ids.append(id)
    articles.append(article)

print("Articles length: ", len(articles_map)) 
prepreprocessed_articles = complete_preprocessing(articles)
bow = create_bow(prepreprocessed_articles)
np.save("ids",  np.array(ids))
np.save("bow_articles", bow)
"""
