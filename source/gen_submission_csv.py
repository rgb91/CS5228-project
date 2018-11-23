import csv
import numpy as np

NAME = 'seq_matcher_submission_n_5_merged_last'
pred_lables = np.load ('submission/submissions/' + str (NAME) + '.npy')
print (pred_lables.shape)
with open ('submission/submissions/' + str (NAME) + '.csv', 'w', newline='') as f:
    writer = csv.writer (f)
    writer.writerow (['article_id', 'category'])
    id = 1
    for label in pred_lables:
        writer.writerow ([id, int (label)])
        id += 1
