from code.create_input_files_part_1 import create_url_file, create_title_file, create_category_file
from code.process_urls import scrape
from code.sentence_similarity import pred

import pandas as pd


#Create required train files
train_file = "data/train_v2.csv"
train_df = pd.read_csv(train_file)
create_title_file(train_df, "data/train_v2_dir/train_v2_title")
create_url_file(train_df, "data/train_v2_dir/train_v2_url")
create_category_file(train_df, "data/train_v2_dir/train_v2_category")


train_file_prev = "data/train.csv"
train_df = pd.read_csv(train_file_prev)
create_title_file(train_df, "data/train_v1_dir/train_v1_title")
create_url_file(train_df, "data/train_v1_dir/train_v1_url")
create_category_file(train_df, "data/train_v1_dir/train_v1_category")


#Create required test files.
test_file = "data/test_v2.csv"
test_df = pd.read_csv(test_file)
create_title_file(test_df, "data/test_dir/test_title")
create_url_file(test_df, "data/test_dir/test_url")


#extract train document from urls
scrape("data/train_v2_dir/train_v2_url", "data/train_v2_dir/")
scrape("data/train_v1_dir/train_v1_url", "data/train_v1_dir/")
scrape("data/test_dir/test_url", "data/test_dir/")

#Get predictions
train_dir, test_dir = "data/train_v2_dir/", "data/test_dir/"
train_title_fp, test_title_fp = train_dir + "train_v2_title", test_dir+"test_title"
cat_path = train_dir + "train_v2_category"
get_preds_np = pred(train_title_fp, test_title_fp, train_dir, test_dir, cat_path)

#Create submission file.
preds = []
for pred in get_preds_np:
    preds.append((int(pred[0]), int(pred[1])))

preds = sorted(preds)
ids, labels = zip(*preds)

df = pd.DataFrame()
df['article_id'] = ids
df['category'] = labels

df.to_csv('sampleSubmission.csv')


