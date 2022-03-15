import os

import pandas as pd
from prettytable import PrettyTable
import tqdm

from source.utils.config import Config

class BioGrid(object):
    
    '''
    This class handles the file IO, data preprecessing, 
    data exploration and basic validation of the BioGrid data: 
     - https://downloads.thebiogrid.org/BioGRID/Release-Archive/

    About:
    This selects protein protein interactions (PPIs) from the biogrid
    dataset, for homo sapiens. The proteins are selected in entrez
    format. It outputs a .csv of PPIs which is in the format
    [interactor A, interactor B]. 

    Public API:
     - process()       Reads in the raw biogrid data, filters it for 
                        human proteins and then selects the enrtez columns     
     - describe()      Summarizes the processed data
     - validate()      Validate the processed data
     - head()          Print the top n rows of the processed data    

    '''

    # Class attrs
    raw_dir = 'data/raw'
    processed_dir = 'data/processed/biogrid'
    raw_suffix = ".txt"    
    
    organism = 'Homo sapiens'
    interactor_a = 'Entrez Gene Interactor A'
    interactor_b = 'Entrez Gene Interactor B'  
    organism_a = 'Organism Name Interactor A'
    organism_b = 'Organism Name Interactor B'

    chunksize=1000

    def __init__(self, config, debug=False, verbose=True):
        '''
        Config file is required to create a unique hash for the
        processed ppi filename - it depends on the biogrid version.
        '''
        self.config = config
        self.debug = debug
        self.verbose = verbose

    def process(self):
        '''
        Process the raw biogrid data
        '''

        # Get the version from the config file 
        version = self._get_conf_version()
        raw_filepath = os.path.join(self.raw_dir, f"{version}{self.raw_suffix}")

        assert os.path.exists(raw_filepath), f"ERROR: BioGrid data version '{version}' not found.\nHave you downloaded the correct version of the data?"

        # Initialise dataframe 
        ppi = pd.DataFrame(columns=[self.interactor_a, self.interactor_b])
        
        # Load and filter the data
        f = self._filter_biogrid        # The filter function to be passed to loader
        ppi = self._load_raw_biogrid(ppi, raw_filepath, f)

        # Validate the dataframe
        is_valid = self.validate(ppi)
        assert is_valid, 'ERROR: Biogrid.process() - processed dataset failed validation, aborting'

        # Save the data
        self._save(ppi)

        return ppi

    def describe(self):
        pass    

    def validate(self, dataframe=None):
        '''
        Validates the dataframe. If no dataframe is passed then it loads 
        the processed dataframe as specified by the config
        '''
        if dataframe is None:
            dataframe = self._load_ppi()

        validation = PrettyTable()
        validation.field_names = ['Criteria', 'Result']

        # Check for NA's
        na_ceck = dataframe.isnull().any()
        has_nas = False
        for header, value in na_ceck.items():
            validation.add_row( [f'Column "{header}" contains null values', value ] )
            if value:
                has_nas = True
        
        # Display validation results
        if self.verbose:
            print(validation)

        return not any([ has_nas ]) # Return True is all checks pass            
        

    def head(self):
        pass       

    def _load_raw_biogrid(self, dataframe, filepath, *filters):
        '''
        Load the raw biogrid data in chunks, passing any supplied filters
        over the chunks then appends filtered data to dataframe.
        Args:
        dataframe       Dataframe to append filtered data to
        filepath        Filepath to raw data
        *filters        Any filters to be applied to the data. 
        '''

        try:

            with pd.read_table(filepath, chunksize=self.chunksize) as data_in:
                for chunk in tqdm.tqdm(data_in):
            
                    for filter in filters:
                        chunk = filter(chunk)

                    frames = [dataframe, chunk]
                    dataframe = pd.concat(frames, ignore_index=True)

                    if self.debug:
                        break

        except Exception as e:

            print(f"ERROR: {e}")

        return dataframe

    def _load_ppi(self):
        '''
        Load a pandas df, with optional test identifier. 
        Uses the hash of the conf version to identify the dataset
        '''
        version = self._get_conf_version()

        filename = f"{self._hash(version)}.csv"

        filepath = os.path.join(self.processed_dir, filename)

        if self.debug:
            filepath = self._make_test_filepath(filepath)

        try:

            dataframe = pd.read_csv(filepath)
        
        except Exception as e: 

            print(f"ERROR: {e}")

        return dataframe                    

    def _save(self, dataframe):
        '''
        Writes out a pandas df, with optional test identifier. 
        Uses the conf version to create a unique hash. 
        '''

        version = self._get_conf_version()

        filename = f"{self._hash(version)}.csv"

        filepath = os.path.join(self.processed_dir, filename)

        if self.debug:
            filepath = self._make_test_filepath(filepath)
        
        try:

            dataframe.to_csv(filepath, index=False)
            if self.verbose:
                print(f'SAVED: {filepath}')
        
        except Exception as e: 

            print(f"ERROR: {e}") 

    def _filter_biogrid(self, chunk):
        '''
        Filters biogrid data by:
         - rows = homo sapiens
         - columns = entrez
        '''

        organism = self.organism
        organism_a = self.organism_a
        organism_b = self.organism_b
        interactor_a = self.interactor_a
        interactor_b = self.interactor_b

        # Filter rows
        human_data = chunk.loc[ (chunk[organism_a] == organism) & 
                                (chunk[organism_b] == organism) ]

        # Filter column
        human_data_entrez = human_data[[interactor_a, interactor_b]]

        return human_data_entrez    

    def _get_conf_version(self):
        '''
        Gets the version of the biogrid data specified in the config
        '''
        conf = self.config.show()
        version = conf['data']['version'] # This should be replaced by a method on config

        return version

    def _hash(self, version):
        '''
        Make a hash based on the version of the biogrid data
        '''
        print(version)
        return hash(str(version))

    def _make_test_filepath(self, filepath):
        '''
        Modify a filepath so that a copy can be used exclusively 
        for testing purposes without risking corrupting project data
        '''

        assert filepath 

        split_fp = filepath.split('.')
        split_fp[0] += '_TEST'

        return '.'.join(split_fp)        
     
if __name__ == "__main__":
    config = Config()
    msig = BioGrid(config, debug=True)
    msig.process()
    