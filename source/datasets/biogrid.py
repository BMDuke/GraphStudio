import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
from prettytable import PrettyTable
from bashplotlib.scatterplot import plot_scatter

from source.utils.config import Config
from source.utils.archive import Archive

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
        ppi = ppi.drop_duplicates()     # Drop any duplicated rows

        # Validate the dataframe
        is_valid = self.validate(ppi)
        assert is_valid, 'ERROR: Biogrid.process() - processed dataset failed validation, aborting'

        # Save the data
        self._save(ppi, add_to_lookup=True)

        return ppi

    def describe(self):
        '''
        Describe the processed data. We will look at the total number of 
        interactions and the distribution of the number of interactions
        individual proteins are in.
        '''
        ppi = self._load_ppi()

        # Entire PPI dataset
        ppi_summary = PrettyTable()
        ppi_summary.field_names = ['Interactions', 'Unique interactors A', 'Unique interactors B', 'Total Unique']
        num_interactions = len(ppi.index)
        unique_a = pd.unique(ppi[self.interactor_a]).size
        unique_b = pd.unique(ppi[self.interactor_b]).size
        unique_total = pd.unique(np.concatenate((ppi[self.interactor_b], ppi[self.interactor_b]))).size
        ppi_summary.add_row([num_interactions, unique_a, unique_b, unique_total])

        summary = ['Count', 'Mean', 'Std', 'Min', 'Max']
        deciles = [f"{i}0%" for i in range(1, 10)]
        percentiles = [f"9{i}%" for i in range(1, 10)]
        permilles = [f"99.{i}%" for i in range(1, 10)]

        # Interactor A - Outgoing connections
        a_summary = PrettyTable()
        a_deciles = PrettyTable()
        a_percentiles = PrettyTable()
        a_permilles = PrettyTable()

        a_summary.field_names = summary
        a_deciles.field_names = deciles
        a_percentiles.field_names = percentiles
        a_permilles.field_names = permilles

        interactor_activity_a = ppi[self.interactor_a].value_counts() # How many time a protein occurs
        activity_description_a = interactor_activity_a.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .91, .92, .93, .94, .95, .96, .97, .98, .99, .991, .992, .993, .994, .995, .996, .997, .998, .999])
        a_stat_summary = [int(activity_description_a[i.lower()]) for i in summary]
        a_stat_decile = [int(activity_description_a[i]) for i in deciles]
        a_stat_percentile = [int(activity_description_a[i]) for i in percentiles]
        a_stat_permille = [int(activity_description_a[i]) for i in permilles]

        a_summary.add_row(a_stat_summary)
        a_deciles.add_row(a_stat_decile)
        a_percentiles.add_row(a_stat_percentile)
        a_permilles.add_row(a_stat_permille)
        
        # Interactor B - Incoming connections
        b_summary = PrettyTable()
        b_deciles = PrettyTable()
        b_percentiles = PrettyTable()
        b_permilles = PrettyTable()

        b_summary.field_names = summary
        b_deciles.field_names = deciles
        b_percentiles.field_names = percentiles
        b_permilles.field_names = permilles

        interactor_activity_b = ppi[self.interactor_b].value_counts() # How many time a protein occurs
        activity_description_b = interactor_activity_b.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .91, .92, .93, .94, .95, .96, .97, .98, .99, .991, .992, .993, .994, .995, .996, .997, .998, .999])
        b_stat_summary = [int(activity_description_b[i.lower()]) for i in summary]
        b_stat_decile = [int(activity_description_b[i]) for i in deciles]
        b_stat_percentile = [int(activity_description_b[i]) for i in percentiles]
        b_stat_permille = [int(activity_description_b[i]) for i in permilles]

        b_summary.add_row(b_stat_summary)
        b_deciles.add_row(b_stat_decile)
        b_percentiles.add_row(b_stat_percentile)
        b_permilles.add_row(b_stat_permille)        
          


        if self.verbose:
            # Context
            print("Interaction pairs should be read Interactor A -> acts on -> Interactor B")
            
            # Summary
            print("\n>>\tPPI dataset summary:\n")
            print(ppi_summary, '\n')

            # Interactor A
            print(">>\tInteractor A summary (outgoing interactions):\n")
            print("\nStatistics of outgoing edges")
            print(a_summary)
            print("\nDeciles of outgoing edges")
            print(a_deciles)
            print("\n Top 10th percentiles of outgoing edges")
            print(a_percentiles)
            print("\n Top 10th permilles of outgoing edges")
            print(a_permilles)
            print('\n')

            # Interactor B
            print(">>\tInteractor B summary (incoming interactions):\n")
            print("\nStatistics of incoming edges")
            print(b_summary)
            print("\nDeciles of incoming edges")
            print(b_deciles)
            print("\n Top 10th percentiles of incoming edges")
            print(b_percentiles)
            print("\n Top 10th permilles of incoming edges")
            print(b_permilles)            
            

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
            print('\n++++++++++++++++ DATASET VALIDATION ++++++++++++++++')
            print(validation)

        return not any([ has_nas ]) # Return True is all checks pass            
        

    def head(self, nrows=10):
        '''
        Display the top n rows of the processed biogrid dataset
        '''
        ppi = self._load_ppi()

        if self.verbose:
            print(ppi.head(nrows))

    def _load_raw_biogrid(self, dataframe, filepath, *filters):
        '''
        Load the raw biogrid data in chunks, passing any supplied filters
        over the chunks then appends filtered data to dataframe.
        Args:
        dataframe       Dataframe to append filtered data to
        filepath        Filepath to raw data
        *filters        Any filters to be applied to the data. 
        '''

        assert os.path.exists(filepath), f"Error {filepath} does not exist"

        try:

            tqdm_disable = False
            if self.debug:
                tqdm_disable = True  
            if self.verbose:  
                print(f"\nLoading BioGrid data set and filtering by:\n Organism: {self.organism}\n Format: {self.interactor_a}\n")
            num_lines = int(subprocess.check_output(f"wc -l {filepath}", shell=True).split()[0]) - 1
            with pd.read_table(filepath, chunksize=self.chunksize) as data_in:
                for chunk in tqdm.tqdm(data_in, total=num_lines/self.chunksize, disable=tqdm_disable):
            
                    for filter in filters:
                        chunk = filter(chunk)

                    frames = [dataframe, chunk]
                    dataframe = pd.concat(frames, ignore_index=True)

                    if self.debug:
                        break

        except Exception as e:

            print(f"ERROR: {e}")
            raise

        return dataframe

    def _load_ppi(self, filepath=None):
        '''
        Load a pandas df, with optional test identifier. 
        Uses the hash of the conf version to identify the dataset
        '''

        if not filepath:

            # Make unique ID
            config = self.config
            archive = Archive(config)
            id = archive.make_id('biogrid', 'version')

            filename = f"{id}.csv"
            filepath = os.path.join(self.processed_dir, filename)

        if self.debug:
            filepath = self._make_test_filepath(filepath)

        try:

            dataframe = pd.read_csv(filepath)
        
        except Exception as e: 

            print(f"ERROR: {e}")

        return dataframe                    

    def _save(self, dataframe, add_to_lookup=False, filepath=None):
        '''
        Writes out a pandas df, with optional test identifier. 
        Uses the conf version to create a unique hash. 
        *Filepath is only for unittesting
        '''

        if not filepath:

            # Make unique ID 
            config = self.config
            archive = Archive(config)
            id = archive.make_id('biogrid', 'version', add_to_lookup=add_to_lookup)

            filename = f"{id}.csv"
            filepath = os.path.join(self.processed_dir, filename)

        

        if self.debug:
            filepath = self._make_test_filepath(filepath)
        
        try:

            dataframe.to_csv(filepath, index=False)
            if self.verbose:
                print(f'SAVED: {filepath}')
        
        except Exception as e: 

            print(f"ERROR: {e}") 

        return filepath

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
    msig = BioGrid(config, debug=False)
    msig.process()
    msig.head()
    msig.describe()
    