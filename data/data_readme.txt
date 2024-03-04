

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


filename (/label_errors_in_train_data)  description
--------------------------------------  -----------

class0_sample_nn.csv - the nn th sample of 20 from class 0 training data
class1_sample_nn.csv - the nn th sample of 20 from class 1 training data

Each of these 40 files (nn = 01 through 20) has 5 columns:
id - integer, unique sample tweet identifier
text - string, content of the tweet
target - binary integer, class of the text where 1 = disaster, 0 = not disaster
questionable_label - binary integer, 0 = no label error, 1 = label error,
                     2 = unclear if label is an error or not
notes - short description justifying a questionable_label value of 1 or 2