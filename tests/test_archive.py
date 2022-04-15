from concurrent.futures import process
from multiprocessing import dummy
import unittest
import yaml
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

from source.etl.msig import MSig
from source.utils.config import Config
from source.utils.archive import Archive

# Configure unittest
unittest.TestLoader.sortTestMethodsUsing = None

processed_data_dir = 'data/processed'
dummy_data_dir = 'tests/dummy_data/archive'
config_name = '__archive_test__'
experiment_name = 'archive_test'

class TestArchive(unittest.TestCase):

    '''
    Testing suite for Archive CLI tool 
    * Set verbose to false so not to disrupt the printing process
    > ✓ _init_lookup() 
    > ✓ _add_to_lookup()
    > ✓ _get_from_lookup()
    > ✓ _remove_from_lookup()
    > ✓ _prune_lookup()  
    > ✓ _make_hash()       
    

    '''    

    @classmethod
    def setUpClass(cls) -> None:     

        # Get the current version of the config file
        # Create a new config file for testing
        # Set parameter values for this file 
        # Create the dummy data dir
        # Propagate with subdirs that are in data/processed

        config = Config()
        cls.current = config.current() # Save current config to restore in tear down
       
        config.new(config_name) # Create new config for testing

        data = {
            'p': 10,
            'q': 35,
            'num_walks': 1,
            'walk_length': 3.146787,
            'window_size': 10,
            'negative_samples': 25
        }

        cls.data = data   # Save the values of this for changes later

        # config.add_experiment('data', experiment_name, )

        # Get all subdirs in the processed data dir
        subdirs = [f.name for f in os.scandir(processed_data_dir) if f.is_dir()]

        # Create the dummy data dir and fill with subdirs
        os.mkdir(dummy_data_dir)        
        for dir in subdirs:
            os.mkdir(os.path.join(dummy_data_dir, dir))  

        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:

        config = Config()
        config.set(cls.current)  
        config.delete(config_name, force=True)

        shutil.rmtree(dummy_data_dir)
        
        return super().tearDownClass()

    def setUp(self) -> None:

        return super().setUp()



    def test_1_init_lookup(self):
        '''
        Testing for:
        > Accuracy:
        > Correctness:
            > Creates a lookup csv in specified location
        > Validity:
        '''

        test_lookup = 'alternative_lookup.csv'
        columns = ['id', 'type', 'version', 'p', 'q', 'walk_length', 'num_walks', 'window_size', 'negative_samples']

        config = Config()
        archive = Archive(config, lookup_fp=test_lookup, alt_dir=dummy_data_dir)

        archive._init_lookup()

        # Check it exists
        p = os.path.join(dummy_data_dir, test_lookup)
        lookup = pd.read_csv(p)

        # ++++ Testing ++++
        self.assertTrue(os.path.exists(p)) # Correctness - location
        self.assertEqual(lookup.columns.tolist(), columns) # Correctness - content
        

    def test_2_add_to_lookup(self):
        '''
        Testing for:
        > Accuracy:
            > The correct information has been added to the lookup
        > Correctness:
        > Validity:
        '''

        test_lookup = 'alternative_lookup.csv'
        id = 'test_id'
        type = 'test'

        config = Config()
        archive = Archive(config, lookup_fp=test_lookup, alt_dir=dummy_data_dir)

        lookup_before = archive._load_lookup()

        params = self.data
        params['version'] = config.current()

        # Add an entry to the lookup
        archive._add_to_lookup(id, type, self.data)

        lookup_after = archive._load_lookup()
        row = lookup_after.iloc[ len(lookup_after.index) -1 ]

        # ++++ Testing ++++
        self.assertFalse(lookup_before.equals(lookup_after)) # Change has been made
        self.assertEqual(len(lookup_after.index), len(lookup_before) + 1) # 1 Item added
        self.assertEqual(row['id'], id) # Accuracy
        self.assertEqual(row['type'], type) # Accuracy
        self.assertEqual(row['p'], params['p']) # Accuracy
        self.assertEqual(row['q'], params['q']) # Accuracy
        self.assertEqual(row['walk_length'], params['walk_length']) # Accuracy
        self.assertEqual(row['num_walks'], params['num_walks']) # Accuracy
        self.assertEqual(row['window_size'], params['window_size']) # Accuracy
        self.assertEqual(row['negative_samples'], params['negative_samples']) # Accuracy


    def test_3_get_from_lookup(self):
        '''
        Testing for:
        > Accuracy:
            > The correct information has been retrieved from lookup
        > Correctness:
        > Validity:
        '''

        test_lookup = 'alternative_lookup.csv'
        id = 'test_id'
        type = 'test'

        config = Config()
        archive = Archive(config, lookup_fp=test_lookup, alt_dir=dummy_data_dir)

        params = self.data
        params['version'] = config.current()

        # Add an entry to the lookup
        archive._add_to_lookup(id, type, self.data)        

        # Retrieve item from lookup
        row = archive._get_from_lookup(id)

        # ++++ Testing ++++
        self.assertEqual(row['id'][0], id) # Accuracy
        self.assertEqual(row['type'][0], type) # Accuracy
        self.assertEqual(row['p'][0], params['p']) # Accuracy
        self.assertEqual(row['q'][0], params['q']) # Accuracy
        self.assertEqual(row['walk_length'][0], params['walk_length']) # Accuracy
        self.assertEqual(row['num_walks'][0], params['num_walks']) # Accuracy
        self.assertEqual(row['window_size'][0], params['window_size']) # Accuracy
        self.assertEqual(row['negative_samples'][0], params['negative_samples']) # Accuracy

    
    def test_4_remove_from_lookup(self):
        '''
        Testing for:
        > Accuracy:
            > The correct record has been removed from the lookup
        > Correctness:
        > Validity:
        '''

        test_lookup = 'alternative_lookup.csv'
        id = 'test_id'
        type = 'test'

        config = Config()
        archive = Archive(config, lookup_fp=test_lookup, alt_dir=dummy_data_dir)

        params = self.data
        params['version'] = config.current()

        # Store initial lookup
        lookup_before = archive._load_lookup()

        # Add an entry to the lookup
        archive._add_to_lookup(id, type, self.data)  
        lookup_added = archive._load_lookup()      

        # Remove entry from lookup
        archive._remove_from_lookup(id)  
        lookup_after = archive._load_lookup()

        # ++++ Testing ++++
        self.assertFalse(lookup_before.equals(lookup_added))
        self.assertFalse(lookup_after.equals(lookup_added))
        self.assertTrue(lookup_before.equals(lookup_after))


    def test_5_prune_lookp(self):
        '''
        Testing for:
        > Accuracy:
            > Only files which have been deleted are pruned
        > Correctness:
            > Files which have been deleted are removed from lookup
            > Files which have not remain in the lookup
        > Validity:
        '''

        test_lookup = 'alternative_lookup.csv'
        id = 'test_id'
        type = 'test'

        config = Config()
        archive = Archive(config, lookup_fp=test_lookup, alt_dir=dummy_data_dir)

        params = self.data
        params['version'] = config.current()

        # Create files in the dummy data directory
        filenames = [f"file-{i}.test" for i in range(10)]
        subdirs = [f.name for f in os.scandir(processed_data_dir) if f.is_dir()]
        for dir in subdirs:
            for file in filenames:
                fp = os.path.join(dummy_data_dir, dir, file)
                fp = pathlib.Path(fp) 
                fp.touch() # Create file
                archive._add_to_lookup(file.split('.')[0], dir, params) # Add to the lookup table
        
        lookup_added = archive._load_lookup()        # Save the look up table

        # Delete all odd files from the directories 
        for dir in subdirs:
            fp = os.path.join(dummy_data_dir, dir)
            files = [f.path for f in os.scandir(fp) if f.is_file() and int(f.name.split('-')[1][0]) % 2 != 0] # If is file and file-1.test is odd
            for file in files:
                os.remove(file)
            files = [f.path for f in os.scandir(fp) if f.is_file()]

        # Prune the lookup table
        archive._prune_lookup()

        # Save the lookup table
        lookup_pruned = archive._load_lookup()        # Save the look up table

        file_ids_added = lookup_added['id'].apply(lambda x: int(x.split('-')[1]))
        file_ids_pruned = lookup_pruned['id'].apply(lambda x: int(x.split('-')[1]))

        file_ids_added_mod_2 = file_ids_added.apply(lambda x: x % 2 == 0)
        file_ids_pruned_mod_2 = file_ids_pruned.apply(lambda x: x % 2 == 0)

        # ++++ Testing ++++
        self.assertGreater(len(lookup_added.index), len(lookup_pruned.index)) # Rows have been deleted
        self.assertFalse(file_ids_added_mod_2.all()) # Old lookup contains files with odd filenames
        self.assertTrue(file_ids_pruned_mod_2.all()) # New lookup contains only files with even filenames
        self.assertGreater(len(file_ids_pruned.index), 0) # New lookup is not empty

        
    def test_6_make_hash(self):
        '''
        Testing for:
        > Accuracy:
            > Hashes are made of the correct information - hard to test
        > Correctness:
            > Hashes are consistent
        > Validity:
        '''

        test_lookup = 'alternative_lookup.csv'
        id = 'test_id'
        type = 'test'

        config = Config()
        archive = Archive(config, lookup_fp=test_lookup, alt_dir=dummy_data_dir)

        params = self.data
        params['version'] = config.current()

        # Default config values
        # PPI layer
        version_1_1 = archive._make_hash('version')
        version_1_2 = archive._make_hash('version')
        # Transition layer
        transition_1_1 = archive._make_hash('version', 'p', 'q')
        transition_1_2 = archive._make_hash('version', 'p', 'q')
        # Walks layer
        walks_1_1 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length')
        walks_1_2 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length')
        # Skipgram layer
        skipgram_1_1 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length', 'window_size', 'negative_samples')
        skipgram_1_2 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length', 'window_size', 'negative_samples')

        del params['version']
        config.edit(config_name, 'version=1.1.101')
        config.add_experiment('data', experiment_name, *[f"{k}={v}" for k, v in params.items()])
        config.set_experiment('data', experiment_name)

        # Updated config values
        # PPI layer
        version_2_1 = archive._make_hash('version')
        version_2_2 = archive._make_hash('version')
        # Transition layer
        transition_2_1 = archive._make_hash('version', 'p', 'q')
        transition_2_2 = archive._make_hash('version', 'p', 'q')
        # Walks layer
        walks_2_1 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length')
        walks_2_2 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length')
        # Skipgram layer
        skipgram_2_1 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length', 'window_size', 'negative_samples')
        skipgram_2_2 = archive._make_hash('version', 'p', 'q', 'num_walks', 'walk_length', 'window_size', 'negative_samples')


        # ++++ Testing ++++
        self.assertEqual(version_1_1, version_1_2) # Consistent
        self.assertEqual(transition_1_1, transition_1_2) # Consistent
        self.assertEqual(walks_1_1, walks_1_2) # Consistent
        self.assertEqual(skipgram_1_1, skipgram_1_2) # Consistent
        self.assertEqual(version_2_1, version_2_2) # Consistent
        self.assertEqual(transition_2_1, transition_2_2) # Consistent
        self.assertEqual(walks_2_1, walks_2_2) # Consistent
        self.assertEqual(skipgram_2_1, skipgram_2_2) # Consistent
        self.assertNotEqual(version_1_1, version_2_1) # Dependent on input
        self.assertNotEqual(transition_1_1, transition_2_1) # Dependent on input
        self.assertNotEqual(walks_1_1, walks_2_1) # Dependent on input
        self.assertNotEqual(skipgram_1_1, skipgram_2_1) # Dependent on input




if __name__ == "__main__":
    TestArchive.main()