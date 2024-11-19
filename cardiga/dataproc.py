#
# File that contains useful functions for data processing of any kind
#

import pickle

def save_pickle(*data,filename='file',append=False):
      filename = filename + '.pickle' if filename.split('.')[-1] != "pickle" else filename
      if append:
        with open(filename, 'rb') as fread: 
            data_read = pickle.load(fread)
        with open(filename, 'wb') as f: 
            pickle.dump(data_read+data,f)      
      else:
        with open(filename, 'wb') as f: 
            pickle.dump(data,f) 
      return

def load_pickle(filename):
    filename = filename + '.pickle' if filename.split('.')[-1] != "pickle" else filename
    with open(filename, 'rb') as f: 
        return pickle.load(f) 
    