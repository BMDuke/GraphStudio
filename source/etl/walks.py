import os
import random
import shutil
import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import networkx as nx
from prettytable import PrettyTable

from source.graphs.n2v import Node2Vec
from source.etl.biogrid import BioGrid
from source.utils.config import Config
from source.utils.archive import Archive

class Walk(object):
    
    '''
    This class is the ETL utility to generate the random walks in the 
    PPI graph. It uses the Node2Vec class to generate the walks, using 
    parameter values from the config manager.

    About:
    The random walks that this class generates are used as 'sentences'
    from which we generate skipgrams in the downstream tasks. This loads
    a processed graph and then samples the neighbourhoods of nodes in 
    the graph. The sampling is controlled by two parameters:
    - num_walks
    - walk_length
    that control the number of times each node is sampled, and the length 
    of each sample respectively. The sampling is performed by the Node2Vec 
    class, this class contributes configuration management and archive 
    management. 

    Public API:
     - process()       Reads in the raw biogrid data, filters it for 
                        human proteins and then selects the enrtez columns     
     - describe()      Summarizes the processed data
     - validate()      Validate the processed data
     - head()          Print the top n rows of the processed data    

    '''

    # Class attrs
    base_dir = 'data/processed'
    source_dir = 'data/processed/transition_probs'
    destination_dir = 'data/processed/walks'

    graph_format = 'pickle'


    def __init__(self, config, debug=False, verbose=True):
        '''
        Config file is required to create an ID for the transition
        probabilities that are calculated. This depends on 
        the biogrid version, p and q. 
        '''
        self.config = config
        self.debug = debug
        self.verbose = verbose
    
    def process(self):
        '''
        This function checks the existance of the resource and any
        resources required to produce it. If the resource doesnt exist
        and its required resources do, it will make the resource. 
        '''

        # Get current values from the config file
        version = self._get_biogrid_version()
        p, q = self._get_p_q_values()
        num_walks, walk_length = self._get_sampling_values()

        # Make resource urls
        transition_fp = self._make_filepath('transition')
        walk_fp = self._make_filepath('walk')

        # Check if resource already exists
        if os.path.exists(walk_fp):
            params = self._dump_config()
            print(f'\nCONFLICT: Resource already exists for:\n')
            for k, v in params:
                print(f"{k:<26}{v}")
            return
        
        # Check required resource exists
        assert os.path.exists(transition_fp), f'ERROR: Resource not found. Have you processed transition probabilities for:\nBiogrid: {version}\np: {p}\n: {q}'
        
        # Load the graph
        n2v = self._load_n2v_graph()
        n2v.set(num_walks=num_walks, walk_length=walk_length)

        # Generate the walks
        directory = self.destination_dir
        filename = self._make_uuid('walk', add_to_lookup=True)
        path = os.path.join(directory, filename)
        try: 

            sample = n2v.generate_walks(filepath=path)
            return sample

        except Exception as e:

            shutil.rmtree(n2v.temp_dir)
            os.remove(walk_fp)
            print(f'ERROR: {e}')
            raise 

    def describe(self):
        ''' 
        things to do:
        - ✓ Data format: pd df
        - ✓ Check resource exists
        - Describe:
            - ✓ Shape
            - ✓ File size
            - ✓ Config info
            - Unique values
            - Counts
        '''

        # Make filepaths
        filepath = self._make_filepath('walk')

        # Get config details
        version = self._get_biogrid_version()
        p, q = self._get_p_q_values()
        num_walks, walk_length = self._get_sampling_values()

        # Check file exists
        filename = os.path.split(filepath)[1]
        
        assert os.path.exists(filepath), f'ERROR: Resource not found: {filename} - Have you generated the walks for biogrid version {version}, p {p}, q {q}, number of walks {num_walks}, walk length {walk_length}?'

        # Load walks
        walks = self._load_walks()
        
        # Get some info about the dataset
        shape = walks.shape
        memory = round(walks.memory_usage(deep=True).sum() / 1024**3, 6)
        size = round(os.stat(filepath).st_size / 1024**3, 6)
        


        # Get unique values and counts
        counts = pd.Series()
        print('Counting node occurences...')
        for i in tqdm.tqdm(range(len(walks.index))):
            for j in range(len(walks.columns)):
                node = walks.iloc[i, j]
                if node not in counts:
                    counts.loc[node] = 0
                counts.loc[node] += 1
        print(counts)

        desc = counts.describe()
        median = counts.median()
        largest_10 = counts.nlargest(n=10)
        smallest_10 = counts.nsmallest(n=10)
        skew = counts.skew()



        
        ## Make the tables
        # Config
        config_table = PrettyTable()
        config_header = ['Biogrid Verion', 'p', 'q', 'Walks per Node', 'Walk Length']
        config_table.field_names = config_header
        config_table.add_row([version, p, q, num_walks, walk_length])

        # Memory
        memory_table = PrettyTable()
        memory_header = ['Disk - GB', 'Virtual - GB']
        memory_table.field_names = memory_header
        memory_table.add_row([size, memory])        

        # Table
        table_table = PrettyTable()
        table_header = ['n Rows', 'n Columns']
        table_table.field_names = table_header
        table_table.add_row([shape[0], shape[1]])

        # Statistics
        stat_table = PrettyTable()
        stat_header = ['mean', 'std', 'median', 'max', 'min', 'skew']
        stat_table.field_names = stat_header
        stat_table.add_row([desc['mean'], desc['std'], median, desc['max'], desc['min'], skew])

        # Largest
        large_table = PrettyTable()
        large_header = ['Rank'] + [i for i in range(1,11)]
        large_table.field_names = large_header
        large_table.add_row(['Node'] + largest_10.index.tolist())
        large_table.add_row(['' for i in range(11)])
        large_table.add_row(['Frequency'] + largest_10.values.tolist())

        # Smallest
        small_table = PrettyTable()
        small_header = ['Rank'] + [i for i in range(1,11)]
        small_table.field_names = small_header
        small_table.add_row(['Node'] + smallest_10.index.tolist())
        small_table.add_row(['' for i in range(11)])
        small_table.add_row(['Frequency'] + smallest_10.values.tolist())        

        ## Print Tables
        if self.verbose:
            print(f'\nConfig Details:')
            print(config_table)
            print(f'\nMemory Details:')
            print(memory_table)
            print(f'\nTable Details:')
            print(table_table)
            print(f'\nNode Sample Frequencies:')
            print(stat_table)
            print(f'\nMost frequent nodes:')
            print(large_table)
            print(f'\nLeast frequent nodes:')
            print(small_table)            



    def validate(self):
        '''
        things to do:
        - Data format: pd df
        - Check resource exists
        - Validate:
            - No NANs
        '''

        # Make filepaths
        filepath = self._make_filepath('walk')

        # Get config details
        version = self._get_biogrid_version()
        p, q = self._get_p_q_values()
        num_walks, walk_length = self._get_sampling_values()

        # Check file exists
        filename = os.path.split(filepath)[1]
        
        assert os.path.exists(filepath), f'ERROR: Resource not found: {filename} - Have you generated the walks for biogrid version {version}, p {p}, q {q}, number of walks {num_walks}, walk length {walk_length}?'

        # Load walks
        walks = self._load_walks()        
        
        contains_nas = walks.isna().any().any()

        # Create the table
        validation_table = PrettyTable()
        header = ['Criteria', 'Result']
        validation_table.field_names = header
        validation_table.add_row(["Walk contains NA's", contains_nas])

        # Print table
        if self.verbose:
            print('Resource Validation')
            print(validation_table)

        return contains_nas

            
            

    def head(self):
        '''
        things to do:
        - Data format: pd df
        - Check resource exists
        '''

        # Make filepaths
        filepath = self._make_filepath('walk')

        # Get config details
        version = self._get_biogrid_version()
        p, q = self._get_p_q_values()
        num_walks, walk_length = self._get_sampling_values()

        # Check file exists
        filename = os.path.split(filepath)[1]
        
        assert os.path.exists(filepath), f'ERROR: Resource not found: {filename} - Have you generated the walks for biogrid version {version}, p {p}, q {q}, number of walks {num_walks}, walk length {walk_length}?'

        # Load walks
        walks = self._load_walks()    

        if self.verbose:
            print(walks.head())
        

    def _get_biogrid_version(self):
        '''
        This handles loading the config file and extracts the 
        current biogrid version 
        '''
        
        config = self.config

        config_values = config.show()
        biogrid_version = config_values['data']['version']

        return biogrid_version

    def _get_p_q_values(self):
        '''
        Handle the config utility and return the current p, q values        
        '''
        config = self.config

        config_values = config.show()
        current_version = config_values['data']['current']

        p = config_values['data']['experiments'][current_version]['p']
        q = config_values['data']['experiments'][current_version]['q']   

        return p, q

    def _get_sampling_values(self):
        '''
        Handle the config utility and return the current values for
        num_walks and walk_length        
        '''
        config = self.config

        config_values = config.show()
        current_version = config_values['data']['current']

        num_walks = config_values['data']['experiments'][current_version]['num_walks']
        walk_length = config_values['data']['experiments'][current_version]['walk_length'] 

        return num_walks, walk_length

    def _make_filepath(self, resource, ext=True):
        '''
        This makes returns the filepath for the requested resource.
        resource:           (str) biogrid, transition or walk
        '''
        mapping = {
            'biogrid': {
                'dir': 'biogrid',
                'extension': 'csv'
            },
            'transition': {
                'dir': 'transition_probs',
                'extension': self.graph_format
            },
            'walk': {
                'dir': 'walks',
                'extension': 'csv'
            }
        }

        # Check exists
        assert resource in mapping.keys(), f'ERROR: Requested resource not found {resource}. Either "biogrid", "transition" or "walk".'

        # Make path to directory 
        path = self.base_dir
        path = os.path.join(path, mapping[resource]['dir'])

        # Generate uuid
        uuid = self._make_uuid(resource)

        # Make filename
        filename = uuid
        if ext:
            filename = f'{uuid}.{mapping[resource]["extension"]}'

        return os.path.join(path, filename)
        

    def _dump_config(self):
        '''
        This returns a list keys and values from the config for data 
        '''
        
        config = self.config

        config_values = config.show()
        current_version = config_values['data']['current']

        dump = [[k, v] for k, v in config_values['data']['experiments'][current_version].items()]

        return dump

    def _load_n2v_graph(self):
        '''
        Just a light wrapper to add some interactivity when saving
        the n2v graph. Writing to JSON can take a while
        '''
        if self.verbose:
            print('Loading graph... ')        

        destination = self._make_filepath('transition', ext=False)           

        n2v = Node2Vec(verbose=self.verbose)
        n2v.load(destination, format=self.graph_format)     

        return n2v

    def _load_walks(self):
        '''
        Load and return the walks dataset
        '''

        filepath = self._make_filepath('walk')

        if self.verbose:
            print(f'\nLoading random walk dataset at \n{filepath}\n')

        try:

            walks = pd.read_csv(filepath, header=None)
            return walks

        except Exception as e:

            print(f'ERROR in Walks._load_walks: {e}')
            raise        

    def _make_uuid(self, resource, add_to_lookup=False):
        '''
        Handles config and archive managers to make uuids for 
        biogrid and transition prob data assets. Based on currrent
        config values.
        Args:
        resource:       (str) either "biogrid", "transition" or 
                        "walk"
        add_to_lookup   (bool) Should this be added to the lookup.
                        Used for write operations
        '''

        config = self.config
        archive = Archive(config)

        if resource == "biogrid":
            uuid = archive.make_id('biogrid', 'version')
        elif resource == 'transition':
            uuid = archive.make_id('transition', 'version', 'p', 'q')
        elif resource == 'walk':
            uuid = archive.make_id('walk', 'version', 'p', 'q', 'num_walks', 'walk_length', add_to_lookup=add_to_lookup)
        else:
            raise ValueError(f"Resource unrecognised: {resource}. Options: ['biogrid', 'transition']")

        return uuid        


                                                   

    

if __name__ == "__main__":
    config = Config()
    walk = Walk(config, debug=False)
    # sample = walk.process()
    # print(sample)
    # walk.describe()
    # walk.validate() 
    # walk.head()
