import re
import string
import contractions as con
import numpy as np
import pandas as pd
from spacy.vocab import Vocab
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
    list(str): each line in list_of_lines is appended with ,[count of urls]
    
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
        glove.twitter.27B.200d.txt file.
    
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
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
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
    Reads in and returns a specified set of embeddings.
    
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
        
    print("Found {} word vectors.".format(len(embedding_index)))
    
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
    prompt_date (tuple or str): tuple of 3 date strings of the form
        'yyyy-mm-dd' or 'latest'
    prompt_log_path (str): path to prompt and context log file
    
    Returns
    dict: dict with 3 keys: 'context', 'prefix_class0' and 'prefix_class1'
    values are also dicts with keys: 'date', 'text', and 'notes'
    """
    df_prompt_data = pd.read_csv(prompt_log_path)
    if prompt_date == "latest":
        # get dates of the latest context and prompt prefixes
        context_date = df_prompt_data.loc[df_prompt_data['prompt_component'] == 'context', 'date'].max()
        prefix_c0_date = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                            (df_prompt_data['class'] == 0), 'date'].max()
        prefix_c1_date = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                            (df_prompt_data['class'] == 1), 'date'].max()
    else:
        context_date = prompt_date[0]
        prefix_c0_date = prompt_date[1]
        prefix_c1_date = prompt_date[2]
    
    context = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'context') & \
                                 (df_prompt_data['date'] == context_date), 'content'].values[0]
    prefix_c0 = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                   (df_prompt_data['class'] == 0) & \
                                   (df_prompt_data['date'] == prefix_c0_date), 'content'].values[0]
    prefix_c1 = df_prompt_data.loc[(df_prompt_data['prompt_component'] == 'prompt_prefix') & \
                                   (df_prompt_data['class'] == 1) & \
                                   (df_prompt_data['date'] == prefix_c1_date), 'content'].values[0]
    return_dict = {
        'context': {'date': context_date,
                    'text': context},
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


def write_aug_tweets(aug_tweets_dict, target_class, out_file):
    """
    
    """
    aug_tweet_lines = ["id,text,target"]
    for key in aug_tweets_dict.keys():
        aug_tweet_line = f"{key},{aug_tweets_dict[key]},{target_class}"
        aug_tweet_lines.append(aug_tweet_line)
    
    success_write = write_lines_to_csv(aug_tweet_lines, out_file)
    
    return(success_write)

