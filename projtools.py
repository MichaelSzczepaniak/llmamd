import re

def fix_spillover_lines(list_of_lines):
    """ Fixes lines read in from a file that should be on a single line.

    Parameters:
    list_of_lines (list): List of strings corresponding to lines read in 
    from a file. Lines are assumed to start with 1 or more digits followed
    by a comma.

    Returns:
    list: List of strings with spillover lines repaired

    """
    fixed_content = []
    start_new_line = True
    fixed_current_line = ""
    
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
                # if both current and next lines start with digit and comma,
                # then current line is on its own line
                fixed_content.append(line.strip())
                start_new_line = True
            else:
                # if current line start with digit and comma but the next line
                # doesn't, assume the next line is a continuation of the current
                fixed_current_line = fixed_current_line + " " + line_next
                start_new_line = False
        else:
            # current line does not start with a digit
            if next_starts_with_digit:
                # if current line doesn't start with a digit, but next one does,
                # assume current line is the last fragment of the previous line
                fixed_content.append(fixed_current_line)
                start_new_line = True
            else:
                # if neither current or next line starts with a digit,
                # assume the next line is a continuation of the current
                fixed_current_line = fixed_current_line + " " + line_next
                start_new_line = False
    
        if i == len(list_of_lines) - 2:
            # next line is last line
            if start_new_line:
                fixed_content.append(line_next.strip())
            else:
                fixed_content.append(fixed_current_line.strip())
    
    return(fixed_content)


def make_url_counts(list_of_lines):
    """ Counts the number of urls in each line of list_of_lines and adds that
    count to the end the line.
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): each line in list_of_lines is appended with ,[count of urls]
    
    """
    return_list = []
    
    header_line = list_of_lines[0]  # add new column headers
    header_line += ",url_count"
    return_list.append(header_line)
    
    for line in list_of_lines[1:]:
        link_count = line.count('http://') + line.count('https://')
        return_list.append(line + "," + str(link_count))
    
    return(return_list)


def replace_urls(list_of_lines):
    """ Replaces urls in each line of fix_url_lines with the text "web link"
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): each line in list_of_lines is appended with ,[count of urls]
    
    """
    fix_url_lines = []
    # replace urls
    for this_line in list_of_lines:
        urls_http = re.findall("http://t.co/[a-zA-Z0-9]{10}", this_line)
        urls_https = re.findall("https://t.co/[a-zA-Z0-9]{10}", this_line)
        if len(urls_http) > 0:
            fix_url_lines.append(re.sub("http://t.co/[a-zA-Z0-9]{10}", "web link", this_line))
        elif len(urls_https) > 0:
            fix_url_lines.append(re.sub("https://t.co/[a-zA-Z0-9]{10}", "web link", this_line))
        else:
            fix_url_lines.append(this_line)
    
    return(fix_url_lines)


def replace_twitter_specials(list_of_lines):
    """ Replaces the @ and # characters with "at " and "hash tag " respectively
        in tweet_string
    
    Args:
    list_of_lines (list(str)): List of strings where each line corresponds to a
    single tweet.
    
    Returns:
    list(str): each element in the list is a tweet string that has had its
    @ and # characters replaced by "at " and "hash tag " respectively

    """
    fixed_content = []
    
    for tweet_string in list_of_lines:
        tweet_string = tweet_string.replace('@', 'at ')
        tweet_string = tweet_string.replace('#', 'hash tag ')
        fixed_content.append(tweet_string)
    
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


def write_lines_to_csv(list_of_lines, file_name = "./data/no_name.csv"):
    with open(file=file_name, mode='w', encoding="utf8", errors='ignore') as f_out:
        for line in list_of_lines:
            f_out.write(line)
            f_out.write('\n')

    return(True)



