import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

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
        file_path = "test_data/Thread_" + str(num_file) + ".dat"
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
                id: others.lower()
            })
        
    #filter all those lines that have Empty as there article text
    return articles
rna = read_news_articles()

def preprocess(article):
    #Remove non-alphanumeric characters
    processed_line = re.sub(r'\W+', ' ', article).strip()
    #Remove all digits
    processed_line = re.sub(r'\w*\d\w*', '', processed_line).strip()
    return processed_line

articles = read_news_articles()
#articles = read_titles()
for id in articles:
    articles[id] = preprocess(articles[id])
articles_list = []
for i in range(len(articles)):
    articles_list.append(articles[str(i)])

train_df = pd.read_csv("train.csv")
labels = train_df.category

"""
Personal Experimentation

label_modified = []
articles_modified = []

print(len(articles))
#Ignoring the 4th class
for i, label in enumerate(labels):
    if label != 4:
        label_modified.append(label)
        articles_modified.append(articles_list[i])
labels = pd.Series(label_modified)
articles_list = articles_modified
"""


tfidf = TfidfVectorizer(sublinear_tf=False, min_df=2, norm='l2', ngram_range=(1,4), stop_words='english')
features = tfidf.fit_transform(articles_list).toarray()

"""
    This section is used to produce the important correlated features per class.
"""

N = 20
for category in range(4):
    print("Category: ", category)
    features_chi2 = chi2(features, labels == category)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
    fourgrams = [v for v in feature_names if len(v.split(' ')) == 4]
    print("# '{}':".format(category))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
    print("  . Most correlated trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:])))
    print("  . Most correlated format fourgrams:\n       . {}".format('\n       . '.join(fourgrams[-N:])))


"""
    This part does a visualization of the features in 2D.
"""


from sklearn.manifold import TSNE

# Sampling a subset of our dataset because t-SNE is computationally expensive
SAMPLE_SIZE = int(len(features)*0.2)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
colors = ['pink', 'green', 'midnightblue', 'orange']#, 'darkgrey']

for category in range(4):
    points = projected_features[(labels[indices] == category).values]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category], label=category)
    plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
                      fontdict=dict(fontsize=15))
    plt.legend()
plt.show()

"""
    This part deals with training all the features.
"""


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

models = [
                GradientBoostingClassifier(n_estimators=500),
                RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0),
                MultinomialNB(),
                LogisticRegression(random_state=0, class_weight="balanced"),
                    ]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
      model_name = model.__class__.__name__
      print("Model_name: ", model_name)
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
          entries.append((model_name, fold_idx, accuracy))
          cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
            size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

print(cv_df.groupby('model_name').accuracy.mean())
