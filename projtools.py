import re
import string
import contractions as con
import numpy as np
import pandas as pd
import random as rand
import spacy as sp
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from openai import OpenAI
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

def fix_spillover_lines(list_of_lines):
    """ Fixes lines read in from a file that should be on a single line.

    Parameters:
    list_of_lines (list): List of strings corresponding to lines read in 
    from a file. Lines are assumed to start with 1 or more digits followed
    by a comma.

    Returns:
    list(str): List of strings with spillover lines repaired

    """
    fixed_content = []
    start_new_line = True
    fixed_current_line = ""
    fixed_lines = 0
    
    for i, line in enumerate(list_of_lines[:-1]):
    
        if i == 0:
            fixed_content.append(line.strip())
            continue  # first line are headers
    
        line_next = list_of_lines[i+1].strip()
        # current or next words start with 1 or more digits followed by a comma?
        current_result = re.search("^[0-9]+[,]", line)
        next_result = re.search("^[0-9]+[,]", line_next)
        current_starts_with_digit = current_result is not None
        next_starts_with_digit = next_result is not None
        
        if start_new_line:
            fixed_current_line = line.strip()
        
        if current_starts_with_digit:
            if next_starts_with_digit:
                # A) If both current and next lines start with a digit and comma
                #    then current line is on its own line.
                fixed_content.append(line.strip())
                start_new_line = True
            else:
                # B) If current line starts with a digit and comma but the next
                #    line doesn't, assume the next line is a continuation of the
                #    current
                fixed_current_line = fixed_current_line + " " + line_next
                start_new_line = False
                fixed_lines += 1
        else:
            # current line does not start with a digit
            if next_starts_with_digit:
                # C) If current line doesn't start with a digit, but next one
                #    does, assume current line is the last fragment of the
                #    previous line
                fixed_content.append(fixed_current_line)
                start_new_line = True
                fixed_lines += 1
            else:
                # D) If neither current or next line starts with a digit,
                #    assume the next line is a continuation of the current
                fixed_current_line = fixed_current_line + " " + line_next
                start_new_line = False
                fixed_lines += 1
    
        if i == len(list_of_lines) - 2:
            # next line is last line
            if start_new_line:
                fixed_content.append(line_next.strip())
            else:
                fixed_content.append(fixed_current_line.strip())
    
    print(f"fix_spillover_lines fixed {fixed_lines} lines...")
    
    return(fixed_content)


def make_url_counts(list_of_lines):
    """ Counts the number of urls in each line of list_of_lines and adds that
    count to the end the line.
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): each line in list_of_lines is appended with ,[count of urls] and
        has the terminating /n (new line) character removed.
    
    """
    return_list = []
    
    header_line = list_of_lines[0].strip()  # add new column headers
    header_line += ",url_count"
    return_list.append(header_line)
    
    for line in list_of_lines[1:]:
        link_count = line.count('http://') + line.count('https://')
        return_list.append(line.strip() + "," + str(link_count))
    
    return(return_list)


def replace_urls(list_of_lines):
    """ Replaces urls in each line of fix_url_lines with the text/token
    "<url>". This token was chose because it maps to an embedding vector in the
    glove.twitter.27B.200d.txt file.
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): each line in list_of_lines has URLs replaced with the <url> token
    
    """
    fix_url_lines = []
    # replace urls
    for this_line in list_of_lines:
        this_line = this_line.strip()
        urls_http = re.findall("http://t.co/[a-zA-Z0-9]{10}", this_line)
        urls_https = re.findall("https://t.co/[a-zA-Z0-9]{10}", this_line)
        if len(urls_http) > 0:
            fix_url_lines.append(re.sub("http://t.co/[a-zA-Z0-9]{10}", "<url>", this_line))
        elif len(urls_https) > 0:
            fix_url_lines.append(re.sub("https://t.co/[a-zA-Z0-9]{10}", "<url>", this_line))
        else:
            fix_url_lines.append(this_line)
    
    return(fix_url_lines)


def expand_contractions(list_of_lines):
    """ Does a crude expansion of contractions like "I'm" into "I am" for each
        word in list_of_lines.
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): where the contractions in each line of list_of_lines is expanded
    
    """
    fixed_lines = []
    for line in list_of_lines:
        expanded_words = []
        for word in line.split():
            expanded_words.append(con.fix(word))
        
        fixed_line = ' '.join(expanded_words)
        fixed_lines.append(fixed_line)
    
    return(fixed_lines)


def replace_twitter_specials(list_of_lines):
    """ Replaces the @ and # characters with "<user> " and "<hashtag> "
        respectively in tweet_string.  These replacement tokens were chose
        because they map to embedding vectors in the
        glove.twitter.27B.100d.txt file.
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): each element in the list is a tweet string that has had its
    @ and # characters replaced by "<user> " and "<hashtag> " respectively and
    has the terminating /n (new line) character removed.

    """
    fixed_content = []
    
    for tweet_string in list_of_lines:
        tweet_string = tweet_string.replace('@', '<user> ')
        tweet_string = tweet_string.replace('#', '<hashtag> ')
        fixed_content.append(tweet_string.strip())
    
    return fixed_content


def spacy_digits_and_stops(df, text_col = 'text', spacy_model="en_core_web_md"):
    """
    Replaces digits with <number> token, removes stop words, removes
    punctuation, lemmatizes remaining tokens and stores them in lower case.
    
    Args:
    df (pandas.core.frame.DataFrame): data with a text_col column holding a
                                      single tweet.
    text_col (str): string column name containing the  tweet in df
    spacy_model (str): spaCy language model used to process text. Default is
                       "en_core_web_md"
    
    Returns:
    list(str): each element in the list is a tweet string that has had its
    @ and # characters replaced by "<user> " and "<hashtag> " respectively and
    has the terminating /n (new line) character removed.
    
    https://stackoverflow.com/questions/47144311/removing-punctuation-using-spacy-attributeerror#71257796
    """
    nlp = sp.load(spacy_model)
    # preserve existing special tokens
    # https://github.com/explosion/spaCy/discussions/12007
    ruler = nlp.add_pipe("entity_ruler", first=True)
    patterns = [{"label": "ORG", "pattern": "<hashtag>"},
                {"label": "PERSON", "pattern": "<user>"},
                {"label": "QUANTITY", "pattern": "<number>"},
                {"label": "LOC", "pattern": "<url>"}]
    ruler.add_patterns(patterns)
    nlp.add_pipe("merge_entities", after="entity_ruler")
    
    # remove the following from the default stop word list
    not_stops = {"you", "on", "not", "from", "was", "but", "your", "all", "no",
                 "when", "now", "more", "over", "some", "first", "full", "down",
                 "may", "only", "last", "many", "never", "any", "everyone",
                 "every", "before", "under", "top", "most", "during", "next",
                 "while", "call", "very", "nothing", "anything", "everything",
                 "sometimes", "serious", "everywhere", "none", "except",
                 "within", "above", "below", "nobody", "afterwards", "anywhere"}
    nlp.Defaults.stop_words -= not_stops  # update default stop words
    
    new_text_col = []
    stops_removed = []
    for index, row in df.iterrows():
        # collect the stop words that are being removed so they can be examined
        # .orth_ property returns the string version of the spaCy token
        # https://stackoverflow.com/questions/49889113/converting-spacy-token-vectors-into-text#57902210
        stop_tokens = [token.orth_.lower() for token in nlp(row[text_col]) if token.is_stop]
        stops_removed.extend(stop_tokens)
        # remove what we consider stop words
        line_tokens = " ".join(token.lemma_.lower() for token in nlp(row[text_col])
                               if not token.is_stop) # and not token.is_punct)
        # normalize consecutive digits
        line_tokens = " ".join(token.lemma_ if not (token.is_digit or token.like_num)
                               else "<number>"
                               for token in nlp(line_tokens))
        # remove punctuation
        line_tokens = " ".join(token.lemma_.lower() for token in nlp(line_tokens)
                               if not token.is_punct)
        new_text_col.append(line_tokens)
    
    df.loc[:, text_col] = new_text_col  # update with revised values
    
    return({'df': df, 'stops_removed': stops_removed})



def replace_with_space(fix_me, removal_chars):
    """ Replaces all the characters in removal_chars by a space in
        the fix_me string
    
    Args:
    fix_me (str): the string we want to replace characters in
    removal_chars (list(str)): list of characters which will be replaced by
        spaces in fix_me
    Returns:
    str: the fix_me string passed in with character in removal_chars replaced
         by spaces

    """
    for char in removal_chars:
        fix_me = fix_me.replace(char, ' ')
    
    return fix_me


##### FINISH ME!!!!!
def replace_with_token(fix_me, removal_chars, normalize_token):
    """ Replaces characters all the characters in removal_chars by a space in
        the fix_me string
    
    Args:
    fix_me (str): the string we want to replace characters in
    removal_chars (list(str)): list of characters which will be replaced by
        spaces in fix_me
    normalize_token (str)
    Returns:
    str: the fix_me string passed in with character in removal_chars replaced
         by spaces

    """
    for char in removal_chars:
        fix_me = fix_me.replace(char, ' ')
    
    return fix_me


def remove_digits_and_punc(list_of_text):
    """ Replaces digits or punctuation in each line of list_of_text with a-zA-Z0-9
    space
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): where digits and punctuation in each line of list_of_lines are
    replaced with spaces
    
    """
    # first, remove the digits
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return_list = [replace_with_space(text_item, digits)
                   for text_item in list_of_text]
    # second, remove punctuation
    return_list = [replace_with_space(text_item, list(string.punctuation))
                   for text_item in return_list]
    
    return(return_list)


def write_lines_to_csv(list_of_lines, file_name = "./data/no_name.csv"):
    """ Write a list of lines to a file.
    
    Args:
    list_of_lines (list(str)): List of strings.
    file_name (str): specifies where the file should be written
    
    Returns:
    boolean : True if the file was written without errors or False if there was

    """
    try:
        with open(file=file_name, mode='w', encoding="utf8", errors='ignore') as f_out:
            for line in list_of_lines:
                f_out.write(line)
                f_out.write('\n')
    except IOError:
        print("Unable to write to disk. Closing file.")
        return(False)

    return(True)


def get_glove_embeds(embed_path = "./embeddings_models/glove.twitter.27B.50d.empty.string.fix.txt",
                     embed_as_np = True):
    """
    Reads in and returns the set of embeddings as dict where the key is word
    or token and value is the embedding vector
    
    Args:
        embed_path(str): path to the glove embeddings we want to read in.
        Default is 50 dimension twitter embeddings with the empty string token
        taken as <> at line 38523
    
    Returns
    dict: a dictionary with keys that are words in the embeddings vocabulary
    and values that are the embedding vectors for those words
    
    """
    t0 = time.time()  # intialize start time
    print(f"Indexing word vectors...")
    # load the embeddings into a dict with keys that are words and
    # values are the embedding vectors for those words
    embedding_index = {}

    with open(embed_path, encoding="utf8") as f:
        for line in f:
            word, coeffs = line.split(maxsplit = 1)
            coeffs = np.fromstring(coeffs, dtype='float', sep=' ')
            if not(embed_as_np):
                coeffs = coeffs.tolist()
            embedding_index[word] = coeffs
    
    t1 = time.time()  # mark finish time
    print(f"Found {len(embedding_index)} word vectors")
    exec_time_mins = (t1 - t0) / 60.
    print(f"Retrieving embeddings took {exec_time_mins: .2f} minutes")
    
    return embedding_index


def get_tweets(tweet_file_path = "./data/train_clean_v03.csv"):
    """
    
    """
    df_tweets = pd.read_csv(tweet_file_path, encoding="utf8")
    tweet_lines = df_tweets['text'].to_list()
    
    return(tweet_lines)



def remove_oov_tokens(vocab, list_of_lines):
    """
    Replaces each out-of-vocabulary (OOV) token with an empty string.
    
    Args:
    vocab (set(srt)):
    list_of_lines (list(str)):
    
    Returns:
    list(str): list_of_lines with the OOV tokens replaced by empty strings
    
    """
    #
    print(f"remove_oov_tokens - size of vocabulary: {len(vocab)}")
    return_list = []
    for line in list_of_lines:
        # assume tokens are separated by spaces
        line_tokens = line.split()
        fixed_line = [t for t in line_tokens if t in vocab]
        # reassemble the tokens
        updated_line = ' '.join(fixed_line)
        return_list.append(updated_line)

    return(return_list)



def save_spacy_nlp():
    """
    
    """
    pass


def get_prompt_setup(prompt_date='latest',
                     prompt_log_path="./data/prompt_log.csv"):
    """
    Gets the data for prompt targeting both classes which are saved in the
    prompt log file.
    
    Args:
    prompt_date (tuple or str): tuple of 4 date strings of the form
        'yyyy-mm-dd' or 'latest'
    prompt_log_path (str): path to prompt and context log file
    
    Returns:
    dict: dictionary with 3 keys: 'context', 'prefix_class0' and 'prefix_class1'
    values are also dicts with keys: 'date', 'text', and 'notes'
    """
    df_prompt_data = pd.read_csv(prompt_log_path)
    if prompt_date == "latest":
        # get dates of the latest context and prompt prefixes
        context0_date = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'context') & \
                                           (df_prompt_data['class'] == 0), 'date'].max()
        
        context1_date = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'context') & \
                                           (df_prompt_data['class'] == 1), 'date'].max()
        
        prefix_c0_date = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                            (df_prompt_data['class'] == 0), 'date'].max()
        
        prefix_c1_date = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                            (df_prompt_data['class'] == 1), 'date'].max()
    else:
        context0_date = prompt_date[0]
        context1_date = prompt_date[1]
        prefix_c0_date = prompt_date[2]
        prefix_c1_date = prompt_date[3]
    
    context0 = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'context') & \
                                  (df_prompt_data['date'] == context0_date) & \
                                  (df_prompt_data['class'] == 0), 'content'].values[0]
    
    context1 = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'context') & \
                                  (df_prompt_data['date'] == context0_date) & \
                                  (df_prompt_data['class'] == 1), 'content'].values[0]
    
    prefix_c0 = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                   (df_prompt_data['class'] == 0) & \
                                   (df_prompt_data['date'] == prefix_c0_date), 'content'].values[0]
                                   
    prefix_c1 = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                   (df_prompt_data['class'] == 1) & \
                                   (df_prompt_data['date'] == prefix_c1_date), 'content'].values[0]
    return_dict = {
        'context0': {'date': context0_date,
                     'text': context0},
        'context1': {'date': context1_date,
                     'text': context1},
        'prefix_class0': {'date': prefix_c0_date,
                          'text': prefix_c0},
        'prefix_class1': {'date': prefix_c1_date,
                          'text': prefix_c1}
    }
    
    return(return_dict)



def get_aug_tweet(context="", prompt_content="", oai_llm="gpt-3.5-turbo"):
    """
    Generates a tweet based on the context, prompt and openAI LLM model being
    used.
    
    Args:
    context (str): string describing the role the LLM takes on as a part of
        generating its response.
    prompt_content (str): the prompt which the LLM uses with the context to
        generate its response
    oai_llm (str): large language model hosted by OpenAI, "gpt-3.5-turbo" (default)
    
    PRECONDITION: openai.OpenAI has been imported and is available in local env
    
    Return:
    str: the text response from the LLM
    
    """
    client = OpenAI()
    completion = client.chat.completions.create(
      model=oai_llm,
      messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": prompt_content}
      ]
    )
    
    aug_gen_tweet = completion.choices[0].message.content
    
    return(aug_gen_tweet)
    

def get_batch_indices(start_index=0, end_index=(4001, 2801), batch_size=200,
                      last_interval=[(4200, 4297), (3100, 3189)]):
    """
    
    
    Return:
    list(tuple(int, int)): 
    
    """
    offset = batch_size - 1
    
    class0_range_starts = tuple(range(start_index, end_index[0], batch_size))
    class1_range_starts = tuple(range(start_index, end_index[1], batch_size))
    
    tweet_batches_class0 = [(start0, start0+offset) for start0 in class0_range_starts]
    tweet_batches_class1 = [(start1, start1+offset) for start1 in class1_range_starts]
    
    tweet_batches_class0.append((4200, 4297))
    tweet_batches_class1.append((3000, 3189))
    
    return_dict = {
        'tweet_batches_class0': tweet_batches_class0,
        'tweet_batches_class1': tweet_batches_class1
    }
    
    return(return_dict)


def generate_tweet_batch(class_label, context_by_class, start_prompt_by_class,
                         batch_start, batch_end,
                         aug_offset=20000,
                         debug=False,
                         tweet_csv_file="./data/train_clean_v03.csv",
                         out_dir="./data/prompts_v05/",
                         prompt_id="v05prompt"):
    """
    Generates a batch of tweets from an LLM and writes them to a file
    
    Args:
    class_label (int): 0 if NOT disaster tweet, 1 if disaster tweet
    context_by_class (str): if class_label is 0, this is the context for the 
        prompt that generates NOT disaster-related tweets.  If class_label is 1,
        this is the context for the prompt that generates disaster-related
        tweets.
    start_prompt_by_class (str): if class_label is 0, this is the start of the 
        prompt that generates NOT disaster-related tweets.  If class_label is 1,
        this is the start of the prompt that generates disaster-related tweets.
    batch_start (int): starting index of the batch
    batch_end (int): ending index of the batch
    aug_offset (int): add this amount to the original id in order to indicate
        that the tweet was generated (i.e. and augmented tweet)
    debug (boolean): If True, bypass call to LLM and use a test string as the 
        augmented tweet response. If False (default), generate augmented tweets 
        from context_by_class, start_prompt_by_class and original tweets.
    tweet_csv_file_file (str): dataframe containing the
        original tweets from which the LLM will generate similar tweets. This
        dataframe is expected to have the following 3 columns:
        id, text and target where
        id is the unique integer identifier for the tweet
        text is the content of the original tweets
        target is 0 if it is not disaster-related or 1 if it is disaster-related
    
    Return:
    dict: with x keys: aug_tweets_dict, class_label, batch_start, barch_end, generation_time
        where aug_tweets_dict is a dictionary...
        created from the tweet_csv_file.
        class is an int which is 0 if the tweet is NOT disaster, 1 if the tweet
        is disaster-related.
        batch_start is an int which is the starting index of the batch
        barch_end is an int which is the ending index of the batch
        generation_time is a float that is execution time to generate the tweets
        in minutes
    """
    t0 = time.time()  # intialize start time
    # read the training data into a dataframe
    df_train = pd.read_csv(tweet_csv_file, index_col='id')
    # filter for the target class specified
    df_by_class = df_train.loc[df_train['target'] == class_label,
                               ['text', 'target']]
    print(f"shape of tweet_csv_file: {df_train.shape}")
    print(f"generating augmented tweets for class: {class_label}")
    print(f"shape of df_by_class: {df_by_class.shape}")
    # get the original tweets for this batch
    df_aug_chunk = df_by_class.iloc[batch_start:(batch_end+1)]
    #
    rows_processed = 0
    aug_tweet_ids = []
    aug_tweets_texts = []
    # aug_tweets_targets = []
    
    for row_index, row in df_aug_chunk.iterrows():
        rows_processed += 1
        prompt_content = start_prompt_by_class + row['text']
        # debug most bypasses call to the LLM
        if debug:
            gen_tweet = "*** augmented tweet text test ***"
        else:
            gen_tweet = get_aug_tweet(context_by_class, prompt_content)
        
        if not(gen_tweet.startswith('"')):
            gen_tweet = '"' + gen_tweet
        if not(gen_tweet.endswith('"')):
            gen_tweet = gen_tweet + '"'
        
        aug_tweets_texts.append(gen_tweet)
        aug_tweet_ids.append(row_index + aug_offset)
        # aug_tweets_targets.append(row['target'])
        if rows_processed % 10 == 0:
            # print(f"processing row {rows_processed} with id {row['id']}")
            print(f"processing row {rows_processed} with id " +
                  f"{row_index + aug_offset}")
    
    dict_aug_tweets = {}
    for i in range(0, len(aug_tweet_ids)):
        dict_aug_tweets[aug_tweet_ids[i]] = aug_tweets_texts[i]
    
    out_file = f"{out_dir}aug_tweets_class{class_label}_{prompt_id}_"
    out_file = f"{out_file}{batch_start:04}_{batch_end:04}.csv"
    
    write_success = write_aug_tweets(dict_aug_tweets, class_label, out_file)
    if write_success:
        print(f"Augmented data file written successfully to {out_file}")
    else:
        print(f"ERROR: Unable to write augmented data file to {out_file}")
    
    t1 = time.time()
    generation_time = (t1 - t0) / 60.
    print(f"time to do {rows_processed} is {generation_time} minutes")
    
    return_dict = {
        'aug_tweets_dict': dict_aug_tweets,
        'class_label': class_label,
        'batch_start': batch_start,
        'batch_end': batch_end,
        'generation_time': generation_time
    }
    
    return(return_dict)



def write_aug_tweets(aug_tweets_dict, target_class, out_file):
    """
    Writes out a dictionary of generated tweets to a CSV file
    
    Args:
    aug_tweets_dict (dict):
    target_class (int) 1 (disaster class) or 0 (not disaster class)
    out_file (str): path of the output file to be written
    
    Returns:
    bool: True is the file was written successfully, False if it was not
    
    """
    aug_tweet_lines = ["id,text,target"]
    for key in aug_tweets_dict.keys():
        aug_tweet_line = f"{key},{aug_tweets_dict[key]},{target_class}"
        aug_tweet_lines.append(aug_tweet_line)
    
    success_write = write_lines_to_csv(aug_tweet_lines, out_file)
    
    return(success_write)


def get_random_samples(df_data,
                       out_file_prefix = "class0_sample_",
                       out_dir = "./data/label_errors_in_train_data/",
                       sample_size=20,
                       number_of_samples=20,
                       random_seed=711):
    """
    Writes out a num_samples number of random samples drawn from df_data after
    creating a 'notes' and 'questionable_label' columns and filling the later
    with 2 (representing 'uncertain label')
    
    
    Args:
    df_data (pandas.core.frame.DataFrame): dataframe from which samples are
      drawn
    sample_sizes (int): the number of random draws to take from df_data
    
    Returns:
    pandas.core.frame.DataFrame with rows that were NOT sampled from df_data
    
    """
    if random_seed is not None:
        rand.seed(random_seed)
    
    
    indices_drawn = []  # holds indices of drawn samples
    for i in range(0, number_of_samples):
        print(f"i={i} | length of df_data BEFORE sample: {df_data.shape[0]}")
        # draw items for this sample
        random_sample = rand.sample(df_data.index.to_list(), sample_size)
        indices_drawn.extend(random_sample)
        df_data_drawn = df_data.loc[df_data.index.isin(random_sample), :]
        # add the cols explaining whether label was questionable or not
        df_data_drawn['questionable_label'] = 2
        df_data_drawn['notes'] = ''
        # write out the items from this sample
        out_file = out_dir + out_file_prefix + str(i+1).zfill(2) + ".csv"
        df_data_drawn.to_csv(out_file, encoding='utf-8')
        # remove the drawn samples
        df_data_after_draws = df_data.loc[~df_data.index.isin(random_sample), :]
        print(f"i={i} | length of df_train_class0 AFTER sample: {df_data_after_draws.shape[0]}")
        df_data = df_data_after_draws

    print(f"rows after {number_of_samples} samples of size {sample_size} taken: {df_data_after_draws.shape[0]}")

    return(df_data_after_draws)

def count_tokens(text_vector,
                 sort_ascending=False,
                 special_tokens="|<user>|<hashtag>|<url>|<number>",
                 vocabulary_size=5327):
    """
    Counts the number of tokens in text_vector based on a set vocabulary size
    
    Args:
    text_vector (pandas.core.series.Series): column in a dataframe that contains
        the text which we want to count token on
    sort_ascending (boolean):
    special_tokens (str): regex pattern for what should be considered tokens in 
        addition to what is considered a token by default
    vocabulary_size (int): the first this number of words with the highest
        frequency/occurrence is considered the vocabulary. Default of 5327
        was selected to drop single instance tokens found in the training set.
    
    Returns:
    pandas.core.frame.DataFrame with 2 columns: token and token_counts which are
    the token and the number of instances of that token found in text_vector
    """
    # add the special tokens to the default tokenization pattern
    token_pat = r"(?u)\b\w\w+\b" + special_tokens
    
    ## add the special tokens to token_pattern parameter so we can preserve them
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
                                 token_pattern = token_pat,
                                 preprocessor = None,
                                 max_features = vocabulary_size)
    data_features = vectorizer.fit_transform(text_vector)
    #
    data_mat = data_features.toarray()
    # dict with keys that are words in the vocabulary, values are the index
    # in the data matrix (data_mat)
    voc_dict = vectorizer.vocabulary_
    word_counts = data_mat.sum(axis=0)

    # turn the matrix of counts into a more readable dataframe
    tokens = []
    token_counts = []
    vocab_tokens = list(voc_dict.keys())
    for token in vocab_tokens:
        tokens.append(token)
        token_counts.append(word_counts[voc_dict[token]])

    df_vocab_counts = pd.DataFrame({'token': tokens,
                                    'token_counts': token_counts})
    df_vocab_counts = df_vocab_counts.sort_values(['token_counts', 'token'],
                                                  ascending=sort_ascending)
    
    return(df_vocab_counts)



def get_tweet_path(tweet_dir, tweet_file_prefix, tweet_class, tweet_prompt,
                   tweet_batch_start, tweet_batch_end, tweet_file_type):
    """
    
    """
    tweet_path = f"{tweet_dir}{tweet_file_prefix}{tweet_class}"
    tweet_path = f"{tweet_path}{tweet_prompt}{tweet_batch_start:04}_"
    tweet_path = f"{tweet_path}{tweet_batch_end:04}{tweet_file_type}"
    
    return(tweet_path)


def get_tweet_outfile(tweet_dir, tweet_file_prefix, tweet_class, tweet_prompt,
                      tweet_file_type):
    """
    
    """
    tweet_out_path = f"{tweet_dir}{tweet_file_prefix}{tweet_class}"
    tweet_out_path = f"{tweet_out_path}{tweet_prompt}{tweet_file_type}"
    
    return(tweet_out_path)


def consolidate_tweet_batches(tweet_batch_dict,
                              tweet_dir='./data/prompts_v05/',
                              tweet_file_prefix='aug_tweets_class',
                              tweet_prompt='_v05prompt_',
                              tweet_file_type='.csv'):
    """
    Reads in a set of files in to dataframes, concatenates them vertically
    (by rows), writes out the consolidated file and returns it to the caller.
    
    
    Args:
    tweet_batch_dict (dict): dictionary with 2 keys: 'tweet_batches_class0' and
                             'tweet_batches_class1'. Values are lists of
                             2-tuples that are the starting and ending indices
                             of the tweet batches to be processed.
    tweet_dir (str): path to the dir where the augmented tweet files reside.
                     Default is './data/prompts_v05/',
    tweet_file_prefix (str): prefix of the augmented tweets file. Default is
                            'aug_tweets_class',
    tweet_prompt (str): designator added to the augmented tweets file to
                        indicate version of the prompt used for their
                        generation. Default is '_v05prompt_',
    tweet_file_type (str): file type of the augmented tweets. Default is '.csv'
    
    
    Returns:
    pandas.core.frame.DataFrame
    
    """
    
    for tweet_class in [0, 1]:
        tweet_batch_class = f"tweet_batches_class{tweet_class}"
        first_tweet_file = \
            get_tweet_path(tweet_dir, tweet_file_prefix,
                           tweet_class, tweet_prompt,
                           tweet_batch_dict[tweet_batch_class][0][0],
                           tweet_batch_dict[tweet_batch_class][0][1],
                           tweet_file_type)
        print(f"First tweet file for class {tweet_class}: {first_tweet_file}")
        df = pd.read_csv(first_tweet_file, header=0, index_col='id',
                         encoding="utf8")
        for tweet_batch in tweet_batch_dict[tweet_batch_class][1:]:
            tweet_file = get_tweet_path(tweet_dir, tweet_file_prefix,
                                        tweet_class, tweet_prompt,
                                        tweet_batch[0], tweet_batch[1],
                                        tweet_file_type)
            print(f"Processessing file: {tweet_file}")
            df_temp = pd.read_csv(tweet_file, header=0, index_col='id',
                                  encoding="utf8")
            df = pd.concat([df, df_temp])
        
        # write the consolidated dataframe
        out_file = get_tweet_outfile(tweet_dir, tweet_file_prefix, tweet_class,
                                     tweet_prompt, tweet_file_type)
        df.to_csv(out_file)
    
    return(df)


def get_vocabulary(path_to_vocab="./data/vocab.txt"):
    """
    
    
    """
    with open(path_to_vocab, encoding="utf8", errors='ignore') as f_vocab:
       vocab = f_vocab.readlines()
    
    clean_vocab = []
    for v in vocab:
        v_clean = v.replace('\n', '')
        clean_vocab.append(v_clean)
    
    return(set(clean_vocab))


def preproccess_pipeline(path_to_input_df_file="./data/train_clean_v03.csv",
                         path_to_output="./data/df_full_pipe.csv",
                         path_to_vocab="./data/vocab.txt",
                         isAugmented=False,
                         isTrain=True):
    """
    Runs the entire preprocessing pipeline from the point immediately after the
    duplicates (both text-target and cross-target) have been removed.
    
    The processed dataframe is written to a file before returning the 
    dataframe to the caller.
    
    Args:
    path_to_input_df_file (str): path to the input csv file
    path_to_output (str): path to write processed data file
    isAugmented (boolean): If False (default), indicates that original data is
        to be processed. If True, indicates that augmented data is to be
        processed.
    isTrain (boolean): if True (default), indicates original training data is
        is to be processed. If False, indicates testing data will be processed
    
    Returns:
    pandas.core.frame.DataFrame having the same fields as the input file, but
    having the text field processed through the pipeline.
    
    """
    df = pd.read_csv(path_to_input_df_file, encoding="utf8")
    
    list_id = df['id'].tolist()
    list_text = df['text'].tolist()  # only field pipeline manipulates
    if isTrain:
        list_target = df['target'].tolist()

    list_urls_fixed = replace_urls(list_text)
    list_twitter_fixed = replace_twitter_specials(list_urls_fixed)
    list_text_fixed = expand_contractions(list_twitter_fixed)
    df_contractions_expanded = pd.DataFrame({'id': list_id,
                                             'text': list_text_fixed})
    dict_df_stops_fixed = spacy_digits_and_stops(df_contractions_expanded)
    df_spacy_fix = dict_df_stops_fixed['df']
    stops_removed = dict_df_stops_fixed['stops_removed']
    spacy_text_fix = df_spacy_fix['text'].tolist()
    # get the vocabulary to the OOV tokens can be removed
    vocab = get_vocabulary(path_to_vocab)
    list_text_fixed = remove_oov_tokens(vocab, spacy_text_fix)

    if isTrain:
        if isAugmented:
            df_full_pipe = pd.DataFrame({'id': list_id,
                                         'text': list_text_fixed,
                                         'target': list_target})
        else: 
            list_keyword = df['keyword'].tolist()
            list_location = df['location'].tolist()
            df_full_pipe = pd.DataFrame({'id': list_id,
                                         'keyword': list_keyword,
                                         'location': list_location,
                                         'text': list_text_fixed,
                                         'target': list_target})
    else:
        list_keyword = df['keyword'].tolist()
        list_location = df['location'].tolist()
        df_full_pipe = pd.DataFrame({'id': list_id,
                                     'keyword': list_keyword,
                                     'location': list_location,
                                     'text': list_text_fixed})

    df_full_pipe.to_csv(path_or_buf=path_to_output,
                        index=False, encoding='utf-8')
    
    return(df_full_pipe)


def word_NN(w, vocab_embeddings, debug=False):
    """
    Finds the word closest to w in the vocabulary that isn't w itself.
    
    Args:
        w(str): string, word to compute nearest neighbor for - must be in a key in vocab_embeddings
        vocab_embeddings(dict): dictionary with keys that are words in the vocabulary
          and values that are d-dimensional numpy array of floats that are the real-
          vector embeddings for each word in the vocabulary
          
    Returns:
        string: the word in the vocabulary that is the closest to this particular word
    
    """
    
    vocab_words = set(vocab_embeddings.keys())
    # check if the word passed in is in the vocabulary
    if not(w in vocab_words):
        print ("Unknown word")
        return
    
    # remove the word we are looking for the nearest neighbor of
    vocab_words.discard(w)
    vocab_words = list(vocab_words)
    
    # get the embedding for passed in word
    w_embedding = vocab_embeddings[w]
    neighbor = 0
    # compute the Euclidean distance between input word and 1st word in vocab
    # here it just used to provide a initial value to compare
    curr_dist = np.linalg.norm(w_embedding - vocab_embeddings[vocab_words[0]])
    # iterate through all the words in the vocabulary and find the 'closest'
    for vocab_word in vocab_words:
        if debug:
            if w_embedding.shape[0] != vocab_embeddings[vocab_word].shape[0]:
                print(f"input word is {w} and had {w_embedding.shape[0]} dims")
                print(f"vocab word is {vocab_word} and has {vocab_embeddings[vocab_word].shape[0]} dims")
        dist = np.linalg.norm(w_embedding - vocab_embeddings[vocab_word])
        if (dist < curr_dist):
            neighbor = vocab_word
            curr_dist = dist
            
    return neighbor


def vectorize_tweet(tweet_string, dict_embs, tokens_to_use=30, pad_token="<>"):
    """
    Creates a one-dimensional vector which is the concatenation of each
    embedding in tweet_string padded to 30 tokens with pad_token
    
    Args:
    tweet_string (str): text of tweet that is to be vectorized, should be
        <= tokens_to_use in length
    dict_embs (dictionary): dictionary with key = words in the vocabulary and
        values = d-dimensional word embedding for that word
    tokens_to_use (int) = number of tokens used to build the results vector
    pad_token (str) = token to be used to pad the results vector up to
        tokens_to_use. Default is empty string token (<>)
    
    Returns:
    numpy.ndarray that is a one-dimensional vector of dtype=float64 with length
    equal to (dimensions of embeddings) x tokens_to_use.  By default, this will
    be              50                  x      30  = 1500
    
    """
    pad_vec = dict_embs[pad_token]
    tokens = tweet_string.split()
    padding_tokens = tokens_to_use - len(tokens)
    token_vec = np.array([])
    for token in tokens:
        token_vec = np.hstack((token_vec, dict_embs[token]))
    # add the padding
    for pad_token in range(padding_tokens):
        token_vec = np.hstack((token_vec, pad_vec))

    return(token_vec)


def make_tweet_feats(list_of_vectors):
    """
    
    Args:
    list_of_vectors (list(numpy.ndarray): list of numpy 1d vectors
    
    Returns:
    numpy.ndarray this is 2d matrix with a row for each element in
    list_of_vectors and columns are the dimensions in each numpy array element
    
    """
    t0 = time.time()  # initialize start time
    v_matrix = list_of_vectors[0]
    # stack each vector underneath the next
    for vec in list_of_vectors[1:]:
        v_matrix = np.vstack((v_matrix, vec))
    t1 = time.time()  # mark end time
    
    exec_time_mins = (t1 - t0) / 60.
    print(f"Building tweet feature matrix took {exec_time_mins: .2f} minutes")
    
    return(v_matrix)


def get_roc_curves(y_tests, y_scores, model_names=None,
                   plot_title='Comparing Model ROCs', colors=None,
                   x_size=8, y_size=8, save_fig=True):
    """
    Creates a plot of ROC curves and their corresponding AUC values for a set of models.

    Args:
          y_tests(np.array(int)): 2-d array where each column are the binary class labels
          (0 or 1) for a particular model. This array MUST have either a single column OR
          the same number of columns as y_scores.
          
          If y_tests is a single column vector, then function assumes that the same y_tests
          values should be a applied to each column in y_scores.
          
          y_scores(np.array(float)): 2-d array where rows are samples and each column are
          the probabilities that each corresponding y_tests value = 1 for a particular model
          
          model_names(list(str)): list of size y_tests.shape[1] = y_scores.shape[1]
          which are the names of the models used to generate each score column. If no model
          names are passed in (default), generic names of the form "model x" will be created
          where x is an integer in [0, y_tests.shape[1])

          colors(list(str)): a list of colors. If None (default) function will use the 10 color
          Tableau pallette: grey or grey, brown, orange, olive, green, cyan, blue, purple, pink, red
          
          x_size (int): horizontal plot scaling
          
          y_size (int): vertical plot scaling
          
          save_fig (boolean): If True (default), saves the figure as a png file
          to the same dir as caller. If False, file is not saved.

    Returns:
        2-tuple: First item is a matplotlib.pyplot object which has a show() method which renders the plot.
        Second item is a dict with keys that are the model_names and values that are the AUC
        of the True Positive Rate vs False Positive Rate (ROC) curve for that model.
                 
    """
    
    # ensure single dim vectors are 1D column vectors so they can be sliced consistently later on
    if len(y_tests.shape) == 1:
        y_tests = y_tests.reshape(-1, 1)
    if len(y_scores.shape) == 1:
        y_scores = y_scores.reshape(-1, 1)
        
    n_models = y_scores.shape[1]
    
    # check shapes of the true labels (y_test) and model-computed probabilities (y_scores)
    if y_tests.shape[1] > 1 and y_scores.shape[1] != y_tests.shape[1]:
        print("get_roc_curves ERROR: ")
        print("y_tests has {} columns, y_scores has {} columns".format(y_tests.shape[1],
                                                                       y_scores))
        return False
    elif y_tests.shape[1] == 1 and y_scores.shape[1] > 1:
        print("DEBUG get_roc_curves: BEFORE expanding y_tests from 1 to {} columns".format(y_scores.shape[1]))
        # If y_tests is a single column vector and n_models > 1, add copies of the single y_tests column
        y_tests = np.reshape(y_tests, (-1, 1))
        print("DEBUG get_roc_curves: BEFORE expansion, y_tests shape = {}".format(y_tests.shape))
        y_expanded = np.copy(y_tests)
        print("DEBUG get_roc_curves: BEFORE expansion, y_expanded shape = {}".format(y_expanded.shape))
        for i in range(n_models-1):
            y_expanded = np.hstack((y_expanded, y_tests))
            print("DEBUG get_roc_curves: DURING expansion, i = {} ".format(i))
            print("DEBUG get_roc_curves: DURING expansion, y_expanded shape = {} ".format(y_expanded.shape))
        y_tests = y_expanded
        print("DEBUG get_roc_curves: AFTER expansion, y_tests columns = {} ".format(y_tests.shape[1]))
    
    print(f"Comparing {n_models} models")
    # If no model names are passed in, create generic names
    if model_names == None:
        model_names = ['model' + str(i) for i in range(n_models)]
    
    plt.figure()
    
    plt.figure(figsize=(x_size, y_size))
    
    lw = 2
    if colors == None:
        # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        colors = ['tab:grey', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple',
                  'tab:green', 'tab:cyan', 'tab:brown', 'tab:olive', 'tab:pink']
    
    color_count = len(colors)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute true pos rate and false pos rate over range of thresholds and AUC for each model
    for i in range(n_models):
        fpr[i], tpr[i], _ = roc_curve(y_tests[:, i], y_scores[:, i])
        roc_auc[model_names[i]] = auc(fpr[i], tpr[i])
    
    # plot reference line: random classifier
    plt.plot([0, 1], [0, 1], color=colors[0], lw=lw, linestyle='--')
    # add traces for each model
    for j in range(0, n_models):
        plt.plot(fpr[j], tpr[j], color=colors[j % color_count + 1],
                 lw=lw, label=model_names[j] + ' (AUC = %0.2f)' % roc_auc[model_names[j]])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(plot_title, fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    
    if save_fig:
        plot_file_name = plot_title.replace(' ', '_') + '.png'
        print(f"saving plot to {plot_file_name}")
        plt.savefig(plot_file_name)
    
    return plt, roc_auc


def nn_train(model, X_train, y_train, X_val, y_val,
             n_epochs=250, batch_size=10, name='nn_model.pth'):
    """
    Trains a neural network model over n_epochs using mini-batch gradient
    descent and returns the best accuracy.  Adapted from:
    https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/
    
    Args:
    model (torch.nn.modules.container.Sequential): pytorch model built with Sequential class
    X_train (): training partition of feature tensor
    y_train (): training partition of labels tensor
    X_val (): validation partition of feature tensor
    y_val (): validation partition of labels tensor
    n_epochs (int): number of passes through the training data to make while training
    batch_size (int): number of samples in a (mini) batch

    Returns:
    Accuracy of the best model
    """
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    # Hold the best model
    best_acc = -np.inf   # init to negative infinity
    best_model = None
    # mark out the start of each batch
    batch_starts = torch.arange(0, len(X_train), batch_size)

    for epoch in range(n_epochs):
        model.train()   # put model in training mode
        with tqdm.tqdm(batch_starts, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for batch in bar:
                # take a batch
                X_batch = X_train[batch:batch+batch_size]
                y_batch = y_train[batch:batch+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()  # reset to zero for each batch, otherwise accumulates
                loss.backward()
                # update the weights
                optimizer.step()
                # sum up the correct predictions
                acc = (y_pred.round() == y_batch).float().mean()
                # report progress
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy after each epoch
        model.eval()
        y_pred = model(X_val)  # validation prediction
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_weights = copy.deepcopy(best_model.state_dict())
    # load and return the best accuracy and model
    model.load_state_dict(best_weights)
    
    return(best_acc, best_model)
