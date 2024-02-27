

filename             description
-------------------  -----------
train.csv            original kaggle training data file: https://www.kaggle.com/competitions/nlp-getting-started/data?select=train.csv
train_clean_v01.csv  train.csv with spillover lines fixed
train_clean_v02.csv  train_clean_v01.csv with text-target duplicates removed 
train_clean_v03.csv  train_clean_v02.csv with cross-target duplicates removed

test.csv             original kaggle testing data file: https://www.kaggle.com/competitions/nlp-getting-started/data?select=test.csv
test_clean_v01.csv   test.csv with spillover lines fixed
test_clean_v02.csv   TBD
test_clean_v03.csv   TBD



Definitions

text-target duplicates  - Tweets that have identical (text, target) tuples
cross-target duplicates - Tweets with the same text string, but with a different target value


