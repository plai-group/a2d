

# general imports
import torch
import itertools
import os
import yaml
import gc
import shutil
import contextlib
import io
import sys
import collections
import glob
import numpy as np

def cleanup_log_dir(log_dir):
    """ Ensures that a file system exists for future logging. """
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def generate_stats_dict(prefix,my_list,if_empty_eval=None):
    """ Creates a dictionary of important statistics for logging. """
    if len(my_list):
        new_dict = {}
        new_dict[prefix+'_mean'] = np.mean(my_list)
        new_dict[prefix+'_med'] = np.median(my_list)
        new_dict[prefix+'_min'] = np.min(my_list)
        new_dict[prefix+'_max'] = np.max(my_list)
        return new_dict
    else:
        new_dict = {}
        new_dict[prefix+'_mean'] = if_empty_eval
        new_dict[prefix+'_med'] = if_empty_eval
        new_dict[prefix+'_min'] = if_empty_eval
        new_dict[prefix+'_max'] = if_empty_eval
        return new_dict

def dict_mean(dict_list):
    """ returns mean of all values in each key of a dictionary. """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def flatten_dict(d, parent_key='', sep='_'):
    """ Converts nested dictionary to a single 'flat' dictionary. """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_network(network_location, model_class):
    """ Loads a nueral network and set to eval mode. """
    # load in the state dictionary containing the model parameters
    model_class.load_state_dict(torch.load(network_location))
    # sets the network settings to ignore drop-out mode etc.
    model_class.eval()
    # return the now loaded network
    return model_class

def removekey(d, key):
    """ deletes dictionary key. """
    r = dict(d)
    del r[key]
    return r

def average_dict(dict_list,suffix='eval_',keys_subset=None):
    """ adds mean values of list based values to dictionary. """
    dict_ = dict_list[0]
    for d in dict_list:
        for k in d.keys():
            dict_[k] += d[k]
    for k in d.keys():
        dict_[k] /= len(dict_list)
    new_d = {}
    if keys_subset is None:
        for k in d.keys():
            new_d[suff_update+k] = new_d[k]
    else:
        for k in keys_subset:
            new_d[suff_update+k] = new_d[k]
    return new_d

def device_mask(device, num_gpu):
    """ replaces specified device with device avaialble on machine. """
    if device =='cpu':
        return 'cpu'
    elif 'cuda:' in device:
        device_num = int(device.split(":",1)[1])
        masked_num = min(device_num,num_gpu-1)
        return 'cuda:' + str(masked_num)

@contextlib.contextmanager
def nostdout():
    """ Helpers to remove text output. """
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def supress_stdout(func):
    """ Helpers to remove text output. """
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper

# function to read meta data
def load_my_yaml(file_location):
    """ Loads yaml file as dictionary. """
    # something to load in yaml file
    with open(file_location) as info:
        info_dict = yaml.load(info, Loader=yaml.FullLoader)
    # return the dictionary
    return info_dict

# function to write out meta data
def write_my_yaml(my_dict, file_location):
    """ save dictionary as yaml file. """
    # write it back
    with open(file_location, 'w') as yaml_file:
        yaml_file.write(yaml.dump(my_dict, default_flow_style=False))
    # nothing to return
    return None

def create_new_directory(path, delete_existing=True):
    """ somethig to create a new directory to store results. """
    if not os.path.exists(path):
        os.makedirs(path)
    elif delete_existing:
        shutil.rmtree(path)           # Removes all the subdirectories!
        os.makedirs(path)
    else:
        raise Exception('directory exists, set delete_existing=True to replace it.')

def print_dict(dct):
    """ Makes dictionary printing prettier. """
    pprint.pprint(dct)

def duplicate_config(new_path, duplicate_path, arg):
    """ Duplicates a config file. """
    # load in the current
    data = load_my_yaml(duplicate_path)
    # create the new directory
    create_new_directory(os.path.dirname(new_path))
    # then write it back out
    write_my_yaml(data, new_path)
    # return the path
    return new_path

def check_config(path,my_dict):
    """ Checks that dictionary and yaml file align. """
    # check that all the keys in my dict are the same as the config
    config = load_my_yaml(path)
    # set True
    aligned = True
    # iterate through and see if it holds
    for key in my_dict.keys():
        if config[key] != my_dict[key]:
            aligned = False
    # now return response
    return aligned

def locate_data_source(database_location, identifiers):
    """ Searches file system for consistent yaml file. """
    # iterate through data base
    data_sources = get_sub_d(database_location)
    for data_source in data_sources:
        yaml_location = database_location+"/"+data_source+"/"+"data.yml"
        file_dict = load_my_yaml(yaml_location)
        if file_dict==identifiers:
            return database_location+"/"+data_source
    # nothing to return
    return None

def get_sub_d(a_dir):
    """ returns all sub-directories. """
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def intersection(lst1, lst2):
    """ returns intersection of two lists. """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def find_satifying_config(database_location, my_dict):
    """ Helps search file system for consistent yaml file. """
    data_sources = get_sub_d(database_location)
    for data_source in data_sources:
        # load in yaml from this data source
        yaml_location = database_location+"/"+data_source+"/"+"data.yml"
        file_dict = load_my_yaml(yaml_location)
        # get intersecting keys
        inter_secting_keys = intersection(list(file_dict.keys()), list(my_dict.keys()))
        inter_secting_dict = {x:my_dict[x] for x in my_dict if my_dict[x] == file_dict[x]}
        if len(inter_secting_dict.keys())==len(inter_secting_keys):
            return database_location+"/"+data_source
    # nothing to return
    return None

def check_torch_mem():
    """ Memory debug helper. """
    print('Current Tensors active in pytorch memory:')
    # prints currently alive Tensors and Variables
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), sys.getsizeof(obj)*1e-9)
        except:
            pass

# convert a list of tuples to a dict
def tup2dict(tup):
    """ Converts a tuple or list of tuples to dictionary. """
    di = {}
    for a, b in tup:
        di[a] = b
    return di

# something to remove keys from a dictionary
def entries_to_remove(entries, the_dict):
    """ deletes set of entries from dictionary. """
    for key in entries:
        if key in the_dict:
            del the_dict[key]
    return the_dict
#
# # generate_dictionary of all combos
# def tests_tree(attributes):
#     # convert dict to key value tuples
#     features = []
#     # iterate through all keys
#     for key in list(attributes.keys()):
#         # init a new sub-folder to enumerate
#         sub_folder = []
#         # now iterate through the values in the sub folder
#         for val in attributes[key]:
#             sub_folder.append((key, val))
#         # now append the sub folder to the list of attributes
#         features.append(sub_folder)
#     # generate all combinations of the features
#     programs_to_eval = list(itertools.product(*features))
#     # now convert all of the elements in this list to dictionaries
#     current_dict = {}
#     for i in range(len(programs_to_eval)):
#         current_dict[i] = tup2dict(programs_to_eval[i])
#     # after this is computed pass it back
#     return current_dict, len(programs_to_eval)
#
# # configure test dictionary
# def configure_test_dict(test_dict):
#     # load in generic configs
#     generic_configs = load_my_yaml('./generic_config.yml')
#     # iterate through all examples
#     for iter in list(test_dict.keys()):
#         # iterate through tests
#         for key in list(generic_configs.keys()):
#             # now iterate through keys and make sure we have all values
#             if key not in list(test_dict[iter].keys()):
#                 test_dict[iter][key] = generic_configs[key]
#     # return the saturated dictionaries
#     return test_dict

def dict_to_hash(config_dict):
    """ converts a dictionary to a hash key. """
    # init config string
    config_string = ""
    # combine all keys and values into single string
    for param, value in sorted(config_dict.items()):
        config_string += param+"="+str(value)+","
    # return the generated key
    return config_string

# get the folder
def get_save_folder(_args, root_dir="expert_examples/", include_task=True):
    """ Finds correct sage folder."""
    # set folders
    if include_task:
        folder = _args.env_type+"/"+_args.env_name+"/"+_args.env_task
    else:
        folder = _args.env_type+"/"+_args.env_name
    # check additional args
    expert_data_path=root_dir+folder
    # where to store stuff
    return expert_data_path


def generate_results_dir(args):
    """ Safely generates file path is generated."""
    # logging
    log_dir = args.log_dir

    try:
        os.makedirs(log_dir)
    except:
        pass
    try:
        os.makedirs(os.path.join(log_dir, args.env_name))
    except:
        pass
    try:
        res_path = os.path.join(log_dir, args.env_name, args.prior+'+'+str(args.seed))
        res = os.makedirs(res_path)
    except:
        pass
    results = open(os.path.join(res_path, 'results.csv'), 'w')
    return results

def mem_leak_debug():
    """ debug helper. """
    print('generating tensors in memory')
    total=0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #print(type(obj), obj.size())
                total += 1
        except:
            pass
    print(total)
