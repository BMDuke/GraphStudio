import os
import random
import shutil
import tqdm
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from prettytable import PrettyTable
import tensorflow_data_validation as tfdv

from source.utils.config import Config
from source.utils.archive import Archive
from source.datasets.text import TextDataset

class Skipgrams(object):
    
    '''
    This class is the ETL utility to generate skipgrams from the biased
    walks dataset. It uses the TextDataset class to perform all the 
    data transformations and and IO. This class handles data archive 
    configuration and management. 

    About:
    The input to this stage if the ETL pipeline are the biased walks generated
    from node2vec. These are treated in the same way as sentences  in a 
    word model. Each node visited on a walk is treated as a tokenised word
    in a sentence and from this, skipgrams are generated. These are then used
    as training data t train the embedding model in downstream processes.
        This stage of the ETL pipeline is parameterised by two variables:
     - negative_samples:        The nunmber of negative skipgrams to include in
                                the dataset for every positive skipgram. This 
                                allows you to optimise computation while learning
     - window_size:             This is the size of the sliding window that is used
                                when generating skipgrams. 

    Public API:
     - process()                Generate the skipgrams and save to disk
     - describe()               Describe the dataset produced
     - validate()               Validate the dataset
     - head()                   Print a sample of the data

    '''

    # Class attrs
    base_dir = 'data/processed'


    def __init__(self, config, debug=False, verbose=True):
        '''
        Instantiate a new instance of the Skipgram ETL tool. It takes a config 
        object as an argument which allows it to retrieve current project 
        parameter values.
        '''
        self.config = config
        self.debug = debug
        self.verbose = verbose


    def process(self):
        '''
        This function is responsible for resource management for the walks and 
        skipgram data assets. If the required resource exists and the target 
        doesnt, then the target resource is generated from the required resource.
        This uses the TextDataset class to perform data transformation. 
        '''

        if self.verbose:
            print('Processing node walk samples into skipgrams...')

        # Get the current config values 
        version = self._get_biogrid_version()
        p, q = self._get_p_q_values()
        num_walks, walk_length = self._get_sampling_values()
        negative_samples, window_size = self._get_skipgram_values()

        # Make resource urls
        walk_fp = self._make_filepath('walk')
        skipgram_fp = self._make_filepath('skipgram', ext=True)

        if self.verbose:
            print('Checking resources...')

        # Check if resource already exists
        if os.path.exists(skipgram_fp):
            params = self._dump_config()
            print(f'\nCONFLICT: Resource already exists for:\n')
            for k, v in params:
                print(f"{k:<26}{v}")
            print()
            raise
        
        # Check required resource exists
        assert os.path.exists(walk_fp), f'ERROR: Resource not found. Have you generated walks for for:\nBiogrid: {version}\np: {p}\nq: {q}\nNumber of walks: {num_walks}\nWalk length: {walk_length}'        

        # Process the skipgrams
        _ = self._make_uuid('skipgram', add_to_lookup=True) # Add the skipgram data asset to archive manager

        try: 

            skipgrams = TextDataset(walk_fp, skipgram_fp, verbose=self.verbose)
            skipgrams.create_skipgram_dataset(window_size=window_size, negative_samples=negative_samples)

            return skipgrams.data()
        
        except Exception as e:

            os.remove(skipgram_fp)
            print(f'ERROR: {e}')
            raise             
        
        

    def describe(self):
        '''
        This function provides a description of the skipgram data assets.
        It describes the current config details, the structure of a trianing
        example, the size of the dataset and any features of the dataset.
        '''

        # Make the filepath and check it exists
        skipgram_fp = self._make_filepath('skipgram')

        assert os.path.exists(skipgram_fp), f'ERROR: resource not found:\n{skipgram_fp}'

        # Get config details
        version = self._get_biogrid_version()
        p, q = self._get_p_q_values()
        num_walks, walk_length = self._get_sampling_values()
        negative_samples, window_size = self._get_skipgram_values()     

        # Measure the size taken on disk
        size = round(os.stat(skipgram_fp).st_size / 1024**3, 6)   

        # Load the data
        walk_fp = ''
        skipgrams = TextDataset(walk_fp, skipgram_fp).data()

        # Get one example - requires drilling into batch
        for i in skipgrams.take(1): # This is a batch
            target = i[0]['target'][0]
            context = i[0]['context'][0]
            label = i[1][0]



        ## Make the tables
        # Config
        biogrid_table = PrettyTable()
        biogrid_header = ['Biogrid Verion']
        biogrid_table.field_names = biogrid_header
        biogrid_table.add_row([version])

        graph_table = PrettyTable()
        graph_header = ['p', 'q', 'Walks per Node', 'Walk Length']
        graph_table.field_names = graph_header
        graph_table.add_row([p, q, num_walks, walk_length])      

        skipgram_table = PrettyTable()
        skipgram_header = ['Window Size', 'Negative Samples']
        skipgram_table.field_names = skipgram_header
        skipgram_table.add_row([window_size, negative_samples])  

        # Disk utilisation
        memory_table = PrettyTable()
        memory_header = ['Disk - GB']
        memory_table.field_names = memory_header
        memory_table.add_row([size])          

        # Example descirption
        example_table = PrettyTable()
        example_header = ['Variable', 'Value']
        example_table.field_names = example_header
        example_table.add_row(['target', target])
        example_table.add_row(['context', context])
        example_table.add_row(['label', label])

        ## Print Tables
        if self.verbose:
            print(f'\nConfig Details:')
            print(biogrid_table)
            print(graph_table)
            print(skipgram_table)
            print(f'\nDisk Utilisation Details:')
            print(memory_table)
            print(f'\nExample Details:')
            print(example_table)


    def validate(self):
        '''
        '''

        # Make the filepath and check it exists
        skipgram_fp = self._make_filepath('skipgram')

        assert os.path.exists(skipgram_fp), f'ERROR: resource not found:\n{skipgram_fp}'

        # Load the data
        walk_fp = ''
        skipgrams = TextDataset(walk_fp, skipgram_fp).data()        
        
        # Schema based example validation 
        if self.verbose:
            print('Generating statistics')
        stats = tfdv.generate_statistics_from_tfrecord(data_location=skipgram_fp)
        if self.verbose:
            print('Inferring schema')        
        schema = tfdv.infer_schema(stats)
        options = tfdv.StatsOptions(schema=schema)
        if self.verbose:
            print('Looking for anomolies')        
        anomalous_example_stats = tfdv.validate_examples_in_tfrecord(
            data_location=skipgram_fp, stats_options=options)
        print(anomalous_example_stats)
        

    def head(self, n=5):
        '''
        Print 1 batch and the top n examples
        '''

        # Make the filepath and check it exists
        skipgram_fp = self._make_filepath('skipgram')

        assert os.path.exists(skipgram_fp), f'ERROR: resource not found:\n{skipgram_fp}'

        # Load the data
        walk_fp = ''
        skipgrams = TextDataset(walk_fp, skipgram_fp).data()        
        
        # Print batch top n examples
        for example in skipgrams.take(1): # This is a batch

            if self.verbose:
                target = example[0]['target']
                context = example[0]['context']
                label = example[1]

                print('\nBatch: ')
                print(f'Targets:\n {target}')
                print(f'Contexts:\n {context}')
                print(f'Labels:\n {label}')

                data = zip(target, context, label)

                print(f'\nHead (n={n}):')
                for i in range(n):
                    t, c, l = next(data)
                    print('Target:\t\t', t)
                    print('Context:\t', c)
                    print('Label:\t\t', l, '\n')     

        
                                                   
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

    def _get_skipgram_values(self):
        '''
        Handle the config utility and return the current values for
        negative_samples and window_size       
        '''

        config = self.config

        config_values = config.show()
        current_version = config_values['data']['current']

        negative_samples = config_values['data']['experiments'][current_version]['negative_samples']
        window_size = config_values['data']['experiments'][current_version]['window_size'] 

        return negative_samples, window_size        

    def _make_filepath(self, resource, ext=True):
        '''
        This makes returns the filepath for the requested resource.
        resource:           (str) biogrid, transition or walk
        '''
        mapping = {
            'walk': {
                'dir': 'walks',
                'extension': 'csv'
            },
            'skipgram': {
                'dir': 'skipgrams',
                'extension': 'tfrecord'
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
            uuid = archive.make_id('walk', 'version', 'p', 'q', 'num_walks', 'walk_length')
        elif resource == 'skipgram':
            uuid = archive.make_id('skipgram', 'version', 'p', 'q', 'num_walks', 'walk_length', 'negative_samples', 'window_size', add_to_lookup=add_to_lookup)            
        else:
            raise ValueError(f"Resource unrecognised: {resource}. Options: ['biogrid', 'transition']")

        return uuid     

    def _dump_config(self):
        '''
        This returns a list keys and values from the config for data 
        '''
        
        config = self.config

        config_values = config.show()
        current_version = config_values['data']['current']

        dump = [[k, v] for k, v in config_values['data']['experiments'][current_version].items()]

        return dump                

    

if __name__ == "__main__":
    conf = Config()
    skipgrams = Skipgrams(conf)
    # skipgrams.process()
    # skipgrams.describe()
    skipgrams.validate()
    # skipgrams.head()