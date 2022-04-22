import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import pandas as pd

class MultiLabelDataset(object):

    '''


    About: 


    Public API:
     - 

    '''


    def __init__(self, verbose=True):
        '''
        '''
        self.verbose = verbose







if __name__ == "__main__":
    ml = MultiLabelDataset()
