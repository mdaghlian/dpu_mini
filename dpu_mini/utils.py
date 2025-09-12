import os
import sys
import string
import random
import subprocess
import time
from collections import defaultdict

opj = os.path.join

def dag_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))    
    return result_str

def dag_str2file(filename, txt):
    file2write = open(filename, 'w')
    file2write.write(txt)
    file2write.close()
def dag_arg_checker(arg2check, idx=None):
    '''arg2check is a string, check if it's a number, return the number if so, otherwise return the string
    Should be able to deal with negative numbers too
    '''
    if idx is not None:
        try: 
            arg2check = arg2check[idx]
        except:
            print(f'Index {idx} not found in {arg2check}')
            print('assuming it is a flag to say something is TRUE --flag ')
            return True
        if arg2check == '':
            print(f'Index {idx} is empty in {arg2check}')
            print('assuming it is a flag to say something is TRUE --flag ')
            return True
        elif '--' in arg2check:
            print(f'Index {idx} is a flag in {arg2check}')
            return True              

            
    # [1] Check if it is a list of arguments
    if ',' in arg2check:
        arg2check_list = arg2check.split(',')
        arg_out = [dag_arg_checker(i) for i in arg2check_list]
        return arg_out
    # [2] Check for common strings
    if arg2check.lower() == 'true':
        return True
    elif arg2check.lower() == 'false':
        return False
    elif arg2check.lower() == 'none':
        return None
    
    # [3] Check for numbers
    if arg2check[0] == '-':
        arg_valence = -1
        arg2check = arg2check[1:]
    else:
        arg_valence = 1

    if arg2check.isdigit():
        arg_out = arg_valence * int(arg2check)
    elif arg2check.replace('.','',1).isdigit():
        arg_out = arg_valence * float(arg2check)                
    else:
        arg_out = arg2check   

    return arg_out
def dag_hyphen_parse(str_prefix, str_in):
    '''dag_hyphen_parse
    checks whether a string has a prefix attached.
    Useful for many BIDS format stuff, and when passing arguments on a lot 
    (sometimes it is not clear whether the prefix will be present or not...)

    E.g., I want to make sure that string "task_name" has the format "task-A" 
    part_task_name = "A"
    full_task_name = "task-A"
    
    dag_hyphen_parse("task", part_task_name)
    dag_hyphen_parse("task", full_task_name)

    Both output -> "task-A"
    
    '''
    if str_prefix in str_in:
        str_out = str_in
    else: 
        str_out = f'{str_prefix}-{str_in}'
    # Check for multiple hyphen
    while '--' in str_out:
        str_out = str_out.replace('--', '-')
    return str_out

def dag_find_file_in_folder(filt, path, return_msg='error', exclude=None, recursive=False, file_limit=9999, inclusive_or=False):
    """get_file_from_substring
    Setup to be compatible with JH linescanning toolbox function (linescanning.utils.get_file_from_substring)
    

    This function returns the file given a path and a substring. Avoids annoying stuff with glob. Now also allows multiple filters 
    to be applied to the list of files in the directory. The idea here is to construct a binary matrix of shape (files_in_directory, nr_of_filters), and test for each filter if it exists in the filename. If all filters are present in a file, then the entire row should be 1. This is what we'll be looking for. If multiple files are found in this manner, a list of paths is returned. If only 1 file was found, the string representing the filepath will be returned. 

    Parameters
    ----------
    filt: str, list
        tag for files we need to select
    path: str
        path to the folder we are searching directory 
        OR a list of strings (files), which will be searched
    return_msg: str, optional
        whether to raise an error (*return_msg='error') or return None (*return_msg=None*). Default = 'error'.
    exclude: str, list, optional:
        Specify string/s to exclude from options. 

    Returns
    ----------
    str
        path to the files containing `string`. If no files could be found, `None` is returned

    list
        list of paths if multiple files were found

    Raises
    ----------
    FileNotFoundError
        If no files usingn the specified filters could be found

    """
    # [1] Setup filters (should be lists): 
    filt_incl = filt
    if isinstance(filt_incl, str):
        filt_incl = [filt_incl]
    filt_excl = exclude
    if (filt_excl!=None) and isinstance(filt_excl, str):
        filt_excl = [filt_excl]

    # [2] List & sort files in directory
    if isinstance(path, str):
        input_is_list = False
        folder_path = path
    elif isinstance(path, list):        
        # The list of files is specified...
        input_is_list = True
        files = path.copy()
    else:
        raise ValueError("Unknown input type; should be string to path or list of files")

    matching_files = []
    files_searched = 0
    if inclusive_or:
        checker = any
    else:
        checker = all # AND 
        
    if input_is_list:   # *** Prespecified list of files ***
        for file_name in files:
            # Check if the file name contains all strings in filt_incl
            if checker(string in file_name for string in filt_incl):                
                # Check if the file name contains any strings in filt_excl, if provided
                if filt_excl is not None and any(string in file_name for string in filt_excl):
                    continue
                
                matching_files.append(file_name)
    
    else:               # *** Walk through folders ***
        for root, dirs, files in os.walk(folder_path):
            if not recursive and root != folder_path:
                break        
            
            for file_name in files:
                files_searched += 1
                file_path = os.path.join(root, file_name)

                # Check the inclusion & exclusion criteria
                file_match = dag_file_name_check(file_name, filt_incl, filt_excl, inclusive_or)
                if file_match:
                    matching_files.append(file_path)

                # Check if the limit has been reached
                if files_searched >= file_limit:
                    sys.exit()

    # Sort the matching files
    match_list = sorted(matching_files)
    
    # Are there any matching files? -> error option
    no_matches = len(match_list)==0
    if no_matches:
        if return_msg == "error":
            raise FileNotFoundError(f"Could not find file with incl {filt_incl}, excluding: {filt_excl}, in {path}")        
        else:
            return None        
    
    # Don't return a list if there is only one element
    if isinstance(match_list, list) & (len(match_list)==1):
        match_list = match_list[0]


    return match_list


def dag_file_name_check(file_name, filt_incl, filt_excl, inclusive=False):
    file_match = False
    if not inclusive: # (AND search)
        # Check if the file name contains all strings in filt_incl
        if all(string in file_name for string in filt_incl):
            file_match = True
    else:
        if any(string in file_name for string in filt_incl):
            file_match = True

    
    # Check if the file name contains any strings in filt_excl
    if filt_excl is not None and any(string in file_name for string in filt_excl):
        file_match = False
    return file_match

rdict = lambda: defaultdict(rdict)


def dag_merge_dicts(a: dict, b: dict, max_depth=3, path=[]):
    '''
    Merge two dictionaries recursively
    Adapted from
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries    
    '''    
    merged_dict = a.copy()  # Create a copy of dictionary 'a' to start with
    for key in b:
        if key in merged_dict:
            if isinstance(merged_dict[key], dict) and isinstance(b[key], dict):
                if len(path) < max_depth:
                    # Recursively merge dictionaries
                    merged_dict[key] = dag_merge_dicts(merged_dict[key], b[key], max_depth, path + [str(key)])
                else:
                    raise Exception('Max depth reached at ' + '.'.join(path + [str(key)]))
            elif merged_dict[key] != b[key]:
                raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            merged_dict[key] = b[key]  # If the key is not in 'merged_dict', add it
    return merged_dict    

def dag_get_cores_used(**kwargs):
    command = kwargs.get('command', None)
    if command is None:
        user_name = os.environ['USER']
        command = f"qstat -u {user_name}"  # Replace with your actual username
    output = subprocess.check_output(command, shell=True).decode('utf-8')
    if output == '':
        return 0

    lines = output.strip().split('\n')
    header = lines[0].split()    
    n_cols = len(lines[1].split())

    count = 0 # sometimes take a second to load...
    while 'qw' in output: # 
        time.sleep(5)
        count += 1
        
        output = subprocess.check_output(command, shell=True).decode('utf-8')
        if output == '':
            return 0    
        if 'Eqw' in output:
            print('EQW')    
            sys.exit()
        print(output)

        lines = output.strip().split('\n')
        header = lines[0].split()    
        if count > 50:
            print('bloop')
            break

    cores_index = header.index('slots')  # Or 'TPN' if 'C' is not available
    cores = 0
    for line in lines[2:]:
        columns = line.split()
        if columns:
            try:
                cores += int(columns[cores_index])
            except ValueError:
                print(f"Error converting {columns[cores_index]} to int")
                continue

    return cores 