'''
Class set up:
> Make a copy of the /data/processed directory to use for testing
> Make a custom config file with particular params

Class tear down:
> Remove testing directory
> Delete testing config file

Test set up:
> 

Test tear down:
> 

Things to test:
> _init_lookup()
> _prune_lookup()
> _make_hash() 
> _add_to_lookup()
> _get_from_lookup()
> _load_lookup()
> _save_lookup()
> _remove_from_lookup()
> make_id()

'''

from concurrent.futures import process
import unittest
import yaml
import os
import pathlib

import numpy as np
import pandas as pd

from source.datasets.msig import MSig
from source.utils.config import Config

# Configure unittest
unittest.TestLoader.sortTestMethodsUsing = None

class TestBiogrid(unittest.TestCase):

    '''
    Testing suite for Biogrid dataset 
    * Set verbose to false so not to disrupt the printing process
    > ✓ 
    > self._get_conf_version()
    > self._filter_biogrid()
    > self._load_raw_biogrid()
    > self._save()
    > self._load_ppi()
    > self.validate()    
    > self.process() - make sure error is thrown if conf version doesnt exist 
    

    '''    

    @classmethod
    def setUpClass(cls) -> None:      

        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        
        return super().tearDownClass()

    def setUp(self) -> None:

        return super().setUp()



    def test_1_get_config_version(self):
        '''
        Testing for:
        > Load does not throw error
        > Accuracy: raw content match for 
            > first line
            > last line
        > Accuracy:
            > Num lines match
        '''

        msig = MSig(debug=True, verbose=False)

        raw = msig._load_raw_msig()  
        first_line = raw[0]
        last_line = raw[-1]

        # ++++ Testing ++++
        self.assertEqual(first_line, self.first['raw']) # Raw content matches
        self.assertEqual(last_line, self.last['raw']) # Raw content matches
        self.assertEqual(len(raw), self.num_rows) # Data is the same size







if __name__ == "__main__":
    TestBiogrid.main()