import re
import difflib
import pandas as pd
import numpy as np


def preprocess(texts):
    processed_text = []
    for text in texts:
        # Remove non-alphanumeric characters
        processed_line = re.sub (r'\W+', ' ', text).strip ()
        # Remove all digits
        processed_line = re.sub (r'\w*\d\w*', '', processed_line).strip ()
        # Lower
        processed_line = processed_line.lower ()
        processed_text.append (processed_line)
        # print(processed_line)
    return np.array (processed_text)


test_df = pd.read_csv ("test_v2.csv", skiprows=[0], header=None)
train_df = pd.read_csv ("train_v2.csv", skiprows=[0], header=None)

test_titles = preprocess (test_df[test_df.columns[1]])
train_titles = preprocess (train_df[train_df.columns[1]])
# train_labels = np.array(train_df[train_df.columns[6]])
train_labels = np.load('train_corrected.npy')[:, [1]]
print(zip(train_labels, train_titles))

test_titles = test_titles[0:20]

for test in test_titles:
    print('===================================================================')
    print('>> ' + test)
    for train_label, train_title in zip(train_labels, train_titles):
        seq = difflib.SequenceMatcher (None, train_title, test)
        similarity = seq.ratio () * 100
        if similarity > 50.0:
            print(str(train_label)+" {:0.2f} ".format(similarity)+' '+train_title)
