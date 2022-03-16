from concurrent.futures import process
import unittest
import os

import numpy as np
import pandas as pd

from source.datasets.biogrid import BioGrid
from source.datasets.msig import MSig
from source.utils.config import Config

# Configure unittest
unittest.TestLoader.sortTestMethodsUsing = None

dummy_data_dir = 'tests/dummy_data'
dummy_data_file = 'biogrid.csv'
dummy_data_test = 'unittesting.csv'

class TestBiogrid(unittest.TestCase):

    '''
    Testing suite for Biogrid dataset 
    * Set verbose to false so not to disrupt the printing process
    > ✓ self._get_conf_version()
    > ✓ self._filter_biogrid()
    > ✓ self._load_raw_biogrid()
    > ✓ self._save()
    > ✓ self._load_ppi()
    > self.validate()    
    > self.process() - make sure error is thrown if conf version doesnt exist 
    

    '''    

    @classmethod
    def setUpClass(cls) -> None:      

        # Set path variables
        cls.raw_dir = 'data/raw'
        cls.processed_dir = 'data/processed/biogrid'
        cls.organism = 'Homo sapiens'
        cls.interactor_a = 'Entrez Gene Interactor A'
        cls.interactor_b = 'Entrez Gene Interactor B'  
        cls.organism_a = 'Organism Name Interactor A'
        cls.organism_b = 'Organism Name Interactor B'
        cls.raw_suffix = ".txt"    

        # Set the chunksize
        cls.chunksize = 1000

        # Get the current config version
        cls.config = Config()

        # Load dummy data
        filepath = os.path.join(dummy_data_dir, dummy_data_file)
        cls.data = pd.read_csv(filepath)

        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:

        conf = Config()
        biogrid = BioGrid(conf, debug=True, verbose=False)

        dest = os.path.join(dummy_data_dir, dummy_data_test)
        dest = biogrid._make_test_filepath(dest)

        os.remove(dest)

        
        return super().tearDownClass()

    def setUp(self) -> None:

        return super().setUp()


    def test_1_get_config_version(self):
        '''
        Testing for:
        > Accuracy: Version matches
        '''

        version = self.config.show()['data']['version']

        conf = Config()
        biogrid = BioGrid(conf, debug=True, verbose=False)

        b_version = biogrid._get_conf_version()

        # ++++ Testing ++++
        self.assertEqual(version, b_version) # Accuracy



    def test_2_filter_biogrid(self):
        '''
        Testing for:
        > Accuracy: 
            > Filtered data is contains only human data
            > Filtered data only contains entrez format - trivially proved by ^
        > Correctness:
            > Mismatches are filtered out
        > Validity:
            > Data contains no NA's
        '''

        data = self.data

        conf = Config()
        biogrid = BioGrid(conf, debug=True, verbose=False)

        b_data = biogrid._filter_biogrid(data)

        # Check both interactors have human origin
        reverse_lookup = pd.DataFrame(columns=[self.organism_a, self.organism_b])
        for row in (range(len(b_data.index))):
            a = b_data.iloc[[row]][self.interactor_a].values[0] # gene ids from filtered data
            b = b_data.iloc[[row]][self.interactor_b].values[0] # gene ids from filtered data
            match = data.loc[ (data[self.interactor_a] == a) & (data[self.interactor_b] == b)] # Find rows containing those gene ids in the original data
            
            concat = [reverse_lookup, match[[self.organism_a, self.organism_b]]]
            reverse_lookup = pd.concat(concat)  

        is_human = (reverse_lookup[self.organism_a] == self.organism) & (reverse_lookup[self.organism_a] == self.organism)
        
        # ++++ Testing ++++
        self.assertTrue(is_human.all()) # Accuracy
        self.assertGreaterEqual(len(data.index), len(b_data.index)) # Mismatches have been filtered out
        self.assertFalse(b_data.isnull().values.any()) # Validity - no NAs


    def test_3_load_raw_biogrid(self):
        '''
        Testing for:
        > Correctness:
            > Raises exception if filepath does not exist
            > Returns dataframe with filtered data
            > Returns dataframe with unfiltered data if no filter is passed
        > Validity:
            > Data contains no NA's
        '''

        empty_dataframe_filtered = pd.DataFrame(columns=[self.interactor_a, self.interactor_b])
        empty_dataframe_unfiltered = pd.DataFrame(columns=self.data.columns)

        conf = Config()
        biogrid = BioGrid(conf, debug=True, verbose=False)

        f = biogrid._filter_biogrid # get the filter function to pass to data loader
        version = biogrid._get_conf_version()
        raw_filepath = os.path.join(self.raw_dir, f"{version}{self.raw_suffix}")

        # Load filtered data
        filtered_data = biogrid._load_raw_biogrid(empty_dataframe_filtered.copy(), raw_filepath, f)
        unfiltered_data = biogrid._load_raw_biogrid(empty_dataframe_unfiltered.copy(), raw_filepath)


        # ++++ Testing ++++
        self.assertGreater(len(filtered_data.index), len(empty_dataframe_filtered.index)) # output is pd.df and contains more rows that an empty table
        self.assertGreater(len(unfiltered_data.index), len(empty_dataframe_unfiltered.index)) # output is pd.df and contains more rows that an empty table
        self.assertFalse(filtered_data.isnull().values.any()) # Validity - no NAs
        self.assertFalse(unfiltered_data.isnull().values.any()) # Validity - no NAs
        with self.assertRaises(AssertionError): # File doesnt exists
            _= biogrid._load_raw_biogrid(empty_dataframe_filtered.copy(), 'null/filepath', f)


    def test_4_save(self):
        '''
        Testing for:
        > Correctness:
            > Saves given data bases to location
        '''

        # Setup 
        data = self.data
        dest = os.path.join(dummy_data_dir, dummy_data_test)

        conf = Config()
        biogrid = BioGrid(conf, debug=True, verbose=False)

        # Save data
        biogrid._save(data, filepath=dest)

        # Reload the data 
        data_in = pd.read_csv(biogrid._make_test_filepath(dest))

        # ++++ Testing ++++
        self.assertTrue(data.equals(data_in))


    def test_5_load_ppi(self):
        '''
        Testing for:
        > Correctness:
            > loads given dataset correctly
        '''

        # Setup 
        data = self.data
        dest = os.path.join(dummy_data_dir, dummy_data_test)

        conf = Config()
        biogrid = BioGrid(conf, debug=True, verbose=False)

        # load data
        data_in = biogrid._load_ppi(filepath=dest)

        # ++++ Testing ++++
        self.assertTrue(data.equals(data_in))  


    def test_6_validate(self):
        '''
        Testing for:
        > Correctness:
            > Correctly identifies invalid data
            > Returns valid for valid data
        '''

        # Setup 
        data = self.data

        conf = Config()
        biogrid = BioGrid(conf, debug=True, verbose=False)

        # Filter the data
        data_filtered = biogrid._filter_biogrid(data)
        
        # Corrupt some data
        data_corrupt = data_filtered.copy()
        data_corrupt.iloc[ 1 ] = np.NaN

        # ++++ Testing ++++
        self.assertTrue(biogrid.validate(data_filtered))
        self.assertFalse(biogrid.validate(data_corrupt))                



if __name__ == "__main__":
    TestBiogrid.main()