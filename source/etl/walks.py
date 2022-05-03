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
from source.etl.etl import ETL

class Walk(ETL):
    
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


    def __init__(self, debug=False, verbose=True):
        '''
        Config file is required to create an ID for the transition
        probabilities that are calculated. This depends on 
        the biogrid version, p and q. 
        '''
        
        super().__init__()

        self.debug = debug
        self.verbose = verbose
    
    def process(self, experiment='current'):
        '''
        This function checks the existance of the resource and any
        resources required to produce it. If the resource doesnt exist
        and its required resources do, it will make the resource. 
        '''

        # Get current values from the config file
        version = self._get_biogrid_version(experiment=experiment)
        p, q = self._get_p_q_values(experiment=experiment)
        num_walks, walk_length = self._get_sampling_values(experiment=experiment)

        # Make resource urls
        transition_fp = self._make_filepath('transition', experiment=experiment)
        walk_fp = self._make_filepath('walk', experiment=experiment)

        # Check if resource already exists
        if os.path.exists(walk_fp):
            params = self._dump_config(experiment=experiment)
            print(f'\nCONFLICT: Resource already exists for:\n')
            for k, v in params:
                print(f"{k:<26}{v}")
            return
        
        # Check required resource exists
        assert os.path.exists(transition_fp), f'ERROR: Resource not found. Have you processed transition probabilities for:\nBiogrid: {version}\np: {p}\n: {q}'
        
        # Load the graph
        n2v = self._load_n2v_graph(experiment=experiment)
        n2v.set(num_walks=num_walks, walk_length=walk_length)

        # Generate the walks
        directory = self.destination_dir
        filename = self._make_uuid('walk', experiment=experiment, add_to_lookup=True)
        path = os.path.join(directory, filename)
        try: 

            sample = n2v.generate_walks(filepath=path)
            return sample

        except Exception as e:

            shutil.rmtree(n2v.temp_dir)
            os.remove(walk_fp)
            print(f'ERROR: {e}')
            raise 

    def describe(self, experiment='current'):
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
        filepath = self._make_filepath('walk', experiment=experiment)

        # Get config details
        version = self._get_biogrid_version(experiment=experiment)
        p, q = self._get_p_q_values(experiment=experiment)
        num_walks, walk_length = self._get_sampling_values(experiment=experiment)

        # Check file exists
        filename = os.path.split(filepath)[1]
        
        assert os.path.exists(filepath), f'ERROR: Resource not found: {filename} - Have you generated the walks for biogrid version {version}, p {p}, q {q}, number of walks {num_walks}, walk length {walk_length}?'

        # Load walks
        walks = self._load_walks(experiment=experiment)
        
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



    def validate(self, experiment='current'):
        '''
        things to do:
        - Data format: pd df
        - Check resource exists
        - Validate:
            - No NANs
        '''

        # Make filepaths
        filepath = self._make_filepath('walk', experiment=experiment)

        # Get config details
        version = self._get_biogrid_version(experiment=experiment)
        p, q = self._get_p_q_values(experiment=experiment)
        num_walks, walk_length = self._get_sampling_values(experiment=experiment)

        # Check file exists
        filename = os.path.split(filepath)[1]
        
        assert os.path.exists(filepath), f'ERROR: Resource not found: {filename} - Have you generated the walks for biogrid version {version}, p {p}, q {q}, number of walks {num_walks}, walk length {walk_length}?'

        # Load walks
        walks = self._load_walks(experiment=experiment)        
        
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

            
            

    def head(self, nrows=5, experiment='current'):
        '''
        things to do:
        - Data format: pd df
        - Check resource exists
        '''

        # Make filepaths
        filepath = self._make_filepath('walk', experiment=experiment)

        # Get config details
        version = self._get_biogrid_version(experiment=experiment)
        p, q = self._get_p_q_values(experiment=experiment)
        num_walks, walk_length = self._get_sampling_values(experiment=experiment)

        # Check file exists
        filename = os.path.split(filepath)[1]
        
        assert os.path.exists(filepath), f'ERROR: Resource not found: {filename} - Have you generated the walks for biogrid version {version}, p {p}, q {q}, number of walks {num_walks}, walk length {walk_length}?'

        # Load walks
        walks = self._load_walks(experiment=experiment)    

        if self.verbose:
            print(walks.head(n=nrows))

        

    def _load_n2v_graph(self, experiment='current'):
        '''
        Just a light wrapper to add some interactivity when saving
        the n2v graph. Writing to JSON can take a while
        '''
        if self.verbose:
            print('Loading graph... ')        

        destination = self._make_filepath('transition', experiment=experiment, ext=False)           

        n2v = Node2Vec(verbose=self.verbose)
        n2v.load(destination, format=self.graph_format)     

        return n2v

    def _load_walks(self, experiment='current'):
        '''
        Load and return the walks dataset
        '''

        filepath = self._make_filepath('walk', experiment=experiment)

        if self.verbose:
            print(f'\nLoading random walk dataset at \n{filepath}\n')

        try:

            walks = pd.read_csv(filepath, header=None)
            return walks

        except Exception as e:

            print(f'ERROR in Walks._load_walks: {e}')
            raise        
    


                                                   

    

if __name__ == "__main__":
    walk = Walk(debug=False)
    # sample = walk.process()
    # print(sample)
    # walk.describe()
    # walk.validate() 
    # walk.head()
