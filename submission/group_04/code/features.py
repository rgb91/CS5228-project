from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
#This is for single layer neural network
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import re
import nltk
from imblearn.over_sampling import SMOTE
nltk.download('averaged_perceptron_tagger')

from bow import complete_preprocessing

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
        file_path = "Thread_keywords_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            lines += fp.read().split("###")
    lines = map(lambda line: line.split("||"), lines)
    articles  = {}
    empty = 0
    ids_set = set()
    for line in lines:
        if len(line) >= 2:
            id, others = line[0], ' '.join(line[1:]).strip().replace(",", " ")
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + others
                #others = titles[id]
            articles.update({
                id: join_lines(complete_preprocessing(others.lower().split("\n")))
            })
            ids_set.add(id)

    #filter all those lines that have Empty as there article text
    return articles

def join_lines(lines):
    sentence = []
    for line in lines:
        sentence += line
    return " ".join(list(set(sentence)))

def read_labels():
    category = {}
    with open("train_v_category", "r") as fp:
        for line in fp:
            id, sent = line.lower().split("\t")
            category.update({
                id: int(sent.strip())
            })
    return category

def preprocess(article):
    #Remove non-alphanumeric characters
    processed_line = re.sub(r'\W+', ' ', article).strip()
    #Remove all digits
    processed_line = re.sub(r'\w*\d\w*', ' ', processed_line).strip()
    #Remove stopwords
    
    return processed_line



ids, articles = zip(*read_news_articles().items())
cat_dict = read_labels()
categories = []

for id in ids:
    categories.append(cat_dict[id])

new_articles = []
new_categories = []


for article, label in zip(articles, categories):
    if label == 4:
        #pass
        new_categories.append(0)
    else:
        new_categories.append(1)

    new_articles.append(article)






categories = new_categories


trainDF = pandas.DataFrame()
trainDF['text'] = articles
trainDF['label'] = categories

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.2)

#m = SMOTE(random_state=2)

#CountVectorizer Models

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1,2), min_df=0.004, max_df=30)
count_vect.fit(trainDF['text'])
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

#sm = SMOTE(random_state=2)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

import numpy as np
#xtrain_tfidf, smote_train_y = sm.fit_sample(xtrain_tfidf, train_y)
#print(train_x.shape)
#print(xtrain_tfidf.shape, smote_train_y.shape)
#print(np.unique(smote_train_y, return_counts=True))


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1,3), max_features=3000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
#sm = SMOTE(random_state=2)
#xtrain_tfidf_ngram, smote_train_y_1 = sm.fit_sample(xtrain_tfidf_ngram, train_y)
#print(train_x.shape)



# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=3000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

xtfidfnc = xtrain_tfidf_ngram_chars.toarray()
xtfidfn =  xtrain_tfidf_ngram.toarray()

combined = []
print xtfidfnc.shape, xtfidfn.shape
for i in range(xtfidfnc.shape[0]):
    combined.append(np.concatenate((xtfidfnc[i], xtfidfn[i]), axis=0))

xvalidtfidfnc = xvalid_tfidf_ngram_chars.toarray()
xvalidtfidfn = xvalid_tfidf_ngram.toarray()

val_combined = []
for i in range(xvalidtfidfnc.shape[0]):
        val_combined.append(np.concatenate((xvalidtfidfnc[i], xvalidtfidfn[i]), axis=0))
combined = np.array(combined)
val_combined = np.array(val_combined)


lda_model = decomposition.LatentDirichletAllocation(n_components=6, learning_method='online', max_iter=100)
lda_model.fit(xtrain_tfidf_ngram_chars)
X_train_lda = lda_model.transform(xtrain_tfidf_ngram_chars)
X_valid_lda = lda_model.transform(xvalid_tfidf_ngram_chars)
"""
print(X_topics.shape)
print X_topics
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
"""

"""
embeddings_index = {}
for i, line in enumerate(open('wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

"""
"""
#Text/NLP based features, Might be useful or not, depends on the datai
trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))
"""

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
    
    return classifier, metrics.accuracy_score(predictions, valid_y)


"""
print("Traning models.....")
# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print "NB, Count Vectors: ", accuracy

from sklearn.neighbors import KNeighborsClassifier

"""
from sklearn.neighbors import KNeighborsClassifier
models = []
# Naive Bayes on Word Level TF IDF Vectors
c, accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print "NB, WordLevel TF-IDF: ", accuracy
models.append((c, xtrain_tfidf))


_, accuracy = train_model(naive_bayes.MultinomialNB(), X_train_lda, train_y, X_valid_lda)
print "LDA model: ", accuracy

_, accuracy = train_model(naive_bayes.MultinomialNB(), combined, train_y, val_combined)
print "NB, COmbined: ", accuracy
# Naive Bayes on Ngram Level TF IDF Vectors
c1, accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "NB, N-Gram Vectors: ", accuracy
models.append((c1,xvalid_tfidf_ngram))
# Naive Bayes on Character Level TF IDF Vectors
c2, accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print "NB, CharLevel Vectors: ", accuracy
models.append((c2, xvalid_tfidf_ngram_chars))
# Linear Classifier on Count Vectors
c3, accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print "LR, Count Vectors: ", accuracy

_, accuracy = train_model(linear_model.LogisticRegression(), combined, train_y, val_combined)
print "LR, Combined: ", accuracy

models.append((c3, xvalid_count))
# Linear Classifier on Word Level TF IDF Vectors
c4, accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print "LR, WordLevel TF-IDF: ", accuracy
models.append((c4,xvalid_tfidf))
# Linear Classifier on Ngram Level TF IDF Vectors
c5, accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "LR, N-Gram Vectors: ", accuracy
models.append((c5,xvalid_tfidf_ngram))
# Linear Classifier on Character Level TF IDF Vectors
c6, accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print "LR, CharLevel Vectors: ", accuracy
models.append((c6, xvalid_tfidf_ngram_chars))

c7, accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print "SVM, Count Vectors: ", accuracy

c8, accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print "SVM, WordLevel TF-IDF: ", accuracy

# SVM on Ngram Level TF IDF Vectors
c9, accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "SVM, N-Gram Vectors: ", accuracy

# RF on Count Vectors
c10, accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=400), xtrain_count, train_y, xvalid_count)
print "RF, Count Vectors: ", accuracy

# RF on Word Level TF IDF Vectors
c11, accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=400), xtrain_tfidf, train_y, xvalid_tfidf)
print "RF, WordLevel TF-IDF: ", accuracy
models.append((c11,xvalid_tfidf))
c12, accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=400), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "RF, N-Gram Vectors: ", accuracy

_, accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=400), combined, train_y, val_combined)
print "RF, Combined: ", accuracy


models.append((c12, xvalid_tfidf_ngram))
# Extereme Gradient Boosting on Count Vectors
c13, accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print "Xgb, Count Vectors: ", accuracy

# Extereme Gradient Boosting on Word Level TF IDF Vectors
c14, accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print "Xgb, WordLevel TF-IDF: ", accuracy
models.append((c14, xvalid_tfidf.tocsc()))

c15, accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram.tocsc(), train_y, xvalid_tfidf_ngram.tocsc())
print "Xgb, WordLevel TF-IDF ngram: ", accuracy
#models.append((c15,xvalid_tfidf_ngram.tocsc()))
# Extereme Gradient Boosting on Character Level TF IDF Vectors
c16, accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print "Xgb, CharLevel Vectors: ", accuracy
#models.append((c16, xvalid_tfidf_ngram_chars.tocsc()))


preds = []
for i, (model, valid) in enumerate(models):
    preds.append(model.predict(valid))

valid_np = np.array(valid_y)
for idx in range(valid_np.shape[0]):
    v_y = valid_np[idx]
    all_pred = []
    for pred in preds:
        all_pred.append(pred[idx])
    print v_y, all_pred


"""

def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
print "NN, Ngram Level TF IDF Vectors",  accuracy


def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
print "NN, Ngram Level TF IDF Vectors",  accuracy
"""
"""

def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print "CNN, Word Embeddings",  accuracy
"""
