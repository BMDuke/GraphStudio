import os
import random
import shutil

import networkx as nx
from prettytable import PrettyTable

from source.graphs.n2v import Node2Vec
from source.datasets.biogrid import BioGrid
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

    graph_format = 'json'


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
        Things to do:
        - Check if version specified by config exists
        - Check required resources exist
        - Load processed graph
        - Configure temp_dir 
        - Create results filepath - add_to_lookup !! 
        - Run generate walks
        - Return sample db
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
            print(f'\nCONFLICT: Resource already exists for:')
            for k, v in params:
                print(f"{k:<6}{v}")
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
            print(f'ERROR: {e}')
            raise 

    def describe(self, n2v=None):
        ''' 
        things to do:
        - Data format: pd df
        - Check resource exists
        - Describe:
            - Shape
            - File size
            - Config info
            - Unique values
            - Counts
        '''
        
        return


    def validate(self, n2v):
        '''
        things to do:
        - Data format: pd df
        - Check resource exists
        - Validate:
            - No NANs
        '''

        return 
            
            

    def head(self, n2v):
        '''
        things to do:
        - Data format: pd df
        - Check resource exists
        '''
    
        return 

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
                'extension': self.graph_format
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

        n2v = Node2Vec()
        n2v.load(destination, format=self.graph_format)     

        return n2v

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
            uuid = archive.make_id('transition', 'version', 'p', 'q', 'num_walks', 'walk_length', add_to_lookup=add_to_lookup)
        else:
            raise ValueError(f"Resource unrecognised: {resource}. Options: ['biogrid', 'transition']")

        return uuid        
                                                   

    

if __name__ == "__main__":
    config = Config()
    walk = Walk(config, debug=False)
    sample = walk.process()
    print(sample)

 