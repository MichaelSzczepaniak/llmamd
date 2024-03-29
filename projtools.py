import re
import string
import contractions as con
import numpy as np
import pandas as pd
import random as rand
import spacy as sp
import time
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI

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
    punctuation and lemmatizes remaining tokens.
    
    Args:
    df ((pandas.core.frame.DataFrame)): data with a text_col column holding a
    single tweet.
    text_col (str): string column name containing the  tweet in df
    spacy_model (str): 
    
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
    nlp.Defaults.stop_words -= not_stops
    
    new_text_col = []
    stops_removed = []
    for index, row in df.iterrows():
        # collect the stop words that are being removed so they can be examined
        # .orth_ property returns the string version of the spaCy token
        # https://stackoverflow.com/questions/49889113/converting-spacy-token-vectors-into-text#57902210
        stop_tokens = [token.orth_.lower() for token in nlp(row[text_col]) if token.is_stop]
        stops_removed.extend(stop_tokens)
    
        line_tokens = " ".join(token.lemma_.lower() for token in nlp(row[text_col])
                               if not token.is_stop
                                  and not token.is_punct)
    
        line_tokens = " ".join(token.lemma_ if not (token.is_digit or token.like_num)
                               else "<number>"
                               for token in nlp(line_tokens))
        # remove punctuation left over from stop word removal
        line_tokens = " ".join(token.lemma_.lower() for token in nlp(line_tokens)
                               if not token.is_punct)
        new_text_col.append(line_tokens)
    
    df.loc[:, text_col] = new_text_col
    
    return({'df': df, 'stops_removed': stops_removed})



def replace_with_space(fix_me, removal_chars):
    """ Replaces characters all the characters in removal_chars by a space in
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


def get_glove_embeds(embed_path = "./embeddings/glove.twitter.27B.200d.txt"):
    """
    Reads in and returns the set of embeddings as dict where the key is word
    or token and value is the embedding vector
    
    Args:
        embed_path(str): path the glove embeddings we want to read in
    
    Returns
    dict: a dictionary with keys that are words in the embeddings vocabulary
    and values that are the embedding vectors for those words
    
    """

    print('Indexing word vectors.')
    # load the embeddings into a dict with keys that are words and
    # values are the embedding vectors for those words
    embedding_index = {}

    with open(embed_path, encoding="utf8") as f:
        for line in f:
            word, coeffs = line.split(maxsplit = 1)
            coeffs = np.fromstring(coeffs, dtype='float', sep=' ')
            embedding_index[word] = coeffs
        
    print(f"Found {len(embedding_index)} word vectors.")
    
    return embedding_index


def get_tweets(tweet_file_path = "./data/train_clean_v03.csv"):
    """
    
    """
    df_tweets = pd.read_csv(tweet_file_path, encoding="utf8")
    tweet_lines = df_tweets['text'].to_list()
    
    return(tweet_lines)



def load_vocab_embeddings(spacy_model, dict_embs, vocab):
    """ 
    
    """
    # "en_core_web_md"
    pass



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
    tweet_batches_class1.append((3100, 3189))
    
    return_dict = {
        'tweet_batches_class0': tweet_batches_class0,
        'tweet_batches_class1': tweet_batches_class1
    }
    
    return(return_dict)


def generate_tweet_batch(class_label, context_by_class, start_prompt_by_class,
                         batch_start, batch_end,
                         debug=False,
                         train_csv_file="./data/train_clean_v03.csv",
                         out_dir="./data/prompts_v05/",
                         prompt_id="v05prompt"):
    """
    Generates a batch of tweets and writes them to a file
    
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
    train_csv_file_file (str): dataframe containing the
        original tweets from which the LLM will generate similar tweets. This
        dataframe is expected to have the following 3 columns:
        id, text and target where
        id is the unique integer identifier for the tweet
        text is the content of the original tweets
        target is 0 if it is not disaster-related or 1 if it is disaster-related
    
    Return:
    dict: with x keys: aug_tweets_dict, class_label, batch_start, barch_end, generation_time
        where aug_tweets_dict is a dictionary...
        created from the train_csv_file.
        class is an int which is 0 if the tweet is NOT disaster, 1 if the tweet
        is disaster-related.
        batch_start is an int which is the starting index of the batch
        barch_end is an int which is the ending index of the batch
        generation_time is a float that is execution time to generate the tweets
        in minutes
    """
    t0 = time.time()  # intialize start time
    # read the training data into a dataframe
    df_train = pd.read_csv(train_csv_file, index_col='id')
    # filter for the target class specified
    df_by_class = df_train.loc[df_train['target'] == class_label,
                               ['text', 'target']]
    print(f"shape of train_csv_file: {df_train.shape}")
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
        if debug:
            gen_tweet = "*** augmented tweet text test ***"
        else:
            gen_tweet = get_aug_tweet(context_by_class, prompt_content)
        
        if not(gen_tweet.startswith('"')):
            gen_tweet = '"' + gen_tweet
        if not(gen_tweet.endswith('"')):
            gen_tweet = gen_tweet + '"'
        aug_tweets_texts.append(gen_tweet)
        # aug_tweet_ids.append(row['id'])
        aug_tweet_ids.append(row_index)
        # aug_tweets_targets.append(row['target'])
        if rows_processed % 10 == 0:
            # print(f"processing row {rows_processed} with id {row['id']}")
            print(f"processing row {rows_processed} with id {row_index}")
    
    dict_aug_tweets = {}
    for i in range(0, len(aug_tweet_ids)):
        dict_aug_tweets[aug_tweet_ids[i]] = aug_tweets_texts[i]
    
    out_file = f"{out_dir}aug_tweets_class{class_label}_{prompt_id}_"
    out_file = f"{out_file}{batch_start:04}_{batch_end-1:04}.csv"
    
    write_success = write_aug_tweets(dict_aug_tweets, class_label, out_file)
    print(f"Augmented data file written successfully: {write_success}")
    
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
                 vocabulary_size=5291):
    """
    
    Returns:
    pandas.core.frame.DataFrame with 2 columns: token and token_counts which are
    the token and the number of instances of that token found in text_vector
    """
    
    token_pat = r"(?u)\b\w\w+\b" + special_tokens
    
    ## add the special tokens to token_pattern parameter so we can preserve them
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
                                 token_pattern = token_pat,
                                 preprocessor = None, max_features = vocabulary_size)
    data_features = vectorizer.fit_transform(text_vector)
    #
    data_mat = data_features.toarray()
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