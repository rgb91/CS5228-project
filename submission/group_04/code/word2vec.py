from bow import complete_preprocessing

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
from bow import complete_preprocessing
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble




def read_titles():
    """
        Function read the titles file and stores it in a list in the format (id, title)
        where id is the docid
    """
    titles = {}
    with open("train_v_title", "r") as fp:
        for line in fp:
            id, sent = line.lower().split("\t")
            titles.update({
                id: sent.strip()
            })
    print("Length of titles: ", len(titles))
    return titles


def read_news_articles():
    """
        Function reads the news articles files : Thread_[0..3]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = 6
    titles =  read_titles()
    lines = []
    for num_file in range(num_files):
        file_path = "Thread_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            lines += fp.read().split("###")
    lines = map(lambda line: line.split("||"), lines)
    articles  = {}
    empty = 0
    ids_set = set()
    for line in lines:
        if len(line) >= 2:
            id, others = line[0], ' '.join(line[1:]).strip()
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + others
            articles.update({
                id: complete_preprocessing(others.lower().split("\n"))
            })
            ids_set.add(id)

    #filter all those lines that have Empty as there article text
    return articles

def read_labels():
    category = {}
    with open("train_v_category", "r") as fp:
        for line in fp:
            id, sent = line.lower().split("\t")
            category.update({
                 id: int(sent.strip())
            })
    return category




def join_lines(lines):
    sentence = []
    for line in lines:
        sentence += line
    print sentence
    return " ".join(list(set(sentence)))

def tag_docs(tokens_list):
    tagged_docs = []
    print "Token list: ", tokens_list
    for index, tokens_sents in enumerate(tokens_list):
        token_list = []
        #print tokens_sents
        for token_sent in tokens_sents:
            token_list += token_sent
        tagged_docs.append(TaggedDocument(words=token_list, tags=[str(index)]))
    return tagged_docs


def create_doc2vec(tag_docs):
    """
        Trains the doc2vec model
    """
    max_epochs = 150                        
    vec_size = 1500                                            
    alpha = 0.025                          
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,                                                                               
                    min_alpha=0.025,
                    min_count=1)
    model.build_vocab(tag_docs)

    for epoch in range(max_epochs):
             print('iteration {0}'.format(epoch))
             model.train(tag_docs,
                         total_examples=len(tag_docs),
                         epochs=model.iter)
             # decrease the learning rate
             model.alpha -= 0.0002         
             # fix the learning rate, no decay
             model.min_alpha = model.alpha
    return model



print("Reading articles")
articles_map = read_news_articles()
ids, articles = [], []
for id, article in articles_map.iteritems():
    ids.append(id)
    articles.append(article)
categories_map = read_labels()

categories = []
for id in ids:
    categories.append(categories_map[id])
    
print("Processing articles")
#prepreprocessed_articles = complete_preprocessing(articles)
print("processing titles")
#preprocessed_titles = complete_preprocessing(titles)
print("augmenting with titles")
#full_representation = augment_with_title(prepreprocessed_articles, preprocessed_titles)
tag_docs = tag_docs(articles)
model = create_doc2vec(tag_docs)
np.save("word2vec", model.docvecs.vectors_docs)
np.save("ids", np.array(ids))
np.save("categories", np.array(categories))
dat = model.docvecs.vectors_docs

import pickle
with open("word2vec.pkl", "wb") as fp:
    pickle.dump(model, fp)



"""
from sklearn.model_selection import train_test_split

trainx, validx, trainy, validy = train_test_split(dat, categories, test_size=0.2, random_state=42)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    if is_neural_net:
        classifier.fit(feature_vector_train, label, epochs=100)
    else:
        classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return classifier, metrics.accuracy_score(predictions, validy)


# Naive Bayes on Word Level TF IDF Vectors
#_, accuracy = train_model(naive_bayes.MultinomialNB(), trainx, trainy, validx)
#print "NB, WordLevel TF-IDF: ", accuracy

# Linear Classifier on Count Vectors
_, accuracy = train_model(linear_model.LogisticRegression(), trainx, trainy, validx)
print "LR, Count Vectors: ", accuracy
"""






