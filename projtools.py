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
                # if current start with digit and comma but the next line doesn't,
                # assume the next line is a continuation of the current
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