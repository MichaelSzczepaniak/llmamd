

filename             description
-------------------  -----------
train.csv            original kaggle training data file: https://www.kaggle.com/competitions/nlp-getting-started/data?select=train.csv
train_clean_v01.csv  train.csv with spillover lines fixed
train_clean_v02.csv  train_clean_v01.csv with text-target duplicates removed 
train_clean_v03.csv  train_clean_v02.csv with cross-target duplicates removed
train_clean_v04.csv  URLs replaced with <url> token
train_clean_v05.csv  @ and # replaced with <user> and <hashtag> tokens
train_clean_v06.csv  expand contractions
train_clean_v07.csv  remove digits, stop words, make everything lower case
train_clean_v08.csv  remove out-of-vocabulary (OOV) words/tokens
train_clean_v09.csv  manually add "<>" to tweets with no value after processing:
                     id = 28, 36, 40, 6407, 7295, 8560 and 9919


test.csv             original kaggle testing data file: https://www.kaggle.com/competitions/nlp-getting-started/data?select=test.csv
test_clean_v01.csv   test.csv with spillover lines fixed
test_clean_v02.csv   skipped
test_clean_v03.csv   skipped
test_clean_v04.csv   URLs replaced with <url> token
test_clean_v05.csv   @ and # replaced with <user> and <hashtag> tokens
test_clean_v06.csv   expand contractions
test_clean_v07.csv   remove digits, stop words, make everything lower case
test_clean_v08.csv   remove out-of-vocabulary (OOV) words/tokens
test_clean_v09.csv   manually add "<>" to tweets with no value after processing:
                     id = 43, 966 and 2436


aug_clean_v07.csv    same processing as t*_clean_v07.csv but on augmented tweets
aug_clean_v08.csv    remove out-of-vocabulary (OOV) words/tokens
                     id = 20611


vocab.txt  all the tokens that are considered as the vocabulary for the project
           These were determined by first finding all the tokens that had a
           mapping in the gloVe twitter embeddings file and then if the token
           occurred more than once in the entire training corpus


Definitions

text-target duplicates  - Tweets that have identical (text, target) tuples.
cross-target duplicates - Tweets with the same text string, but with a different
                          target value.  These were all removed from the
                          training set because ground truth could not be
                          established.


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

data files generated from vectorize_tweets.ipynb but are not in this repo
(because they are too large) ---------------------------------------------------

The "feats_" prefix in the four files below means that columns are a dimensional
embedding value for a token in the tweet.  For example, the values in row 8
correspond to the 8th tweet of a particular dataset (aug = augmented,
train_train = training portion of the original training data, train_test =
testing portion of the original training data).  In the cleaned training data,
this tweet is:

"on top hill fire wood"

Assuming the default 50d twitter embeddings are being used, then each block of
50 columns represent an embedding vector for a token in a tweet.  The first 3
values of the 50d embedding vector for the word "top" is:

[0.3779, -0.614, 0.47629, ...]

So the value in row 5 (1-based counting), column 52 would be -0.614 because the
the first 50 values would be the embedding for the word "on".


feats_matrix_aug.txt  7485 rows where each row is a vectorized tweet padded to
                      30 tokens where each token is represented by a 50d GloVe
                      twitter embedding and the empty string is used as the
                      padding token.  1500 cols are the tweets padded to 30
                      tokens which are each converted to a 50d GloVe embedding
feats_matrix_train_train.txt  80% of the original training data used to train each
                              model, same vectorization as feats_matrix_aug.txt,
                              5988 rows, 1500 columns
feats_matrix_train_test.txt   20% of the original training data used to test each
                              model, same vectorization as feats_matrix_aug.txt,
                              1497 rows, 1500 columns
feats_matrix_test.txt         unlabeled test data provided by kaggle to test submissions
                              3263 rows, 1500 columns
labels_aug.txt          labels corresponding to feats_matrix_aug.txt
labels_train_train.txt  labels corresponding to feats_matrix_train_train.txt
labels_train_test.txt   labels corresponding to feats_matrix_train_test.txt
