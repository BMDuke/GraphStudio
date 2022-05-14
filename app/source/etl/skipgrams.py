import os
import random
import shutil
import tqdm
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from prettytable import PrettyTable
import numpy as np

from source.utils.validation import validate_dataset_slower
from source.etl.etl import ETL
from source.datasets.text import TextDataset

class Skipgrams(ETL):
    
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


    def __init__(self, debug=False, verbose=True):
        '''
        Instantiate a new instance of the Skipgram ETL tool. It takes a config 
        object as an argument which allows it to retrieve current project 
        parameter values.
        '''

        super().__init__()

        self.debug = debug
        self.verbose = verbose


    def process(self, experiment='current'):
        '''
        This function is responsible for resource management for the walks and 
        skipgram data assets. If the required resource exists and the target 
        doesnt, then the target resource is generated from the required resource.
        This uses the TextDataset class to perform data transformation. 
        '''

        if self.verbose:
            print('Processing node walk samples into skipgrams...')

        # Get the current config values 
        version = self._get_biogrid_version(experiment=experiment)
        p, q = self._get_p_q_values(experiment=experiment)
        num_walks, walk_length = self._get_sampling_values(experiment=experiment)
        negative_samples, window_size = self._get_skipgram_values(experiment=experiment)

        # Make resource urls
        walk_fp = self._make_filepath('walk', experiment=experiment)
        skipgram_fp = self._make_filepath('skipgram', ext=True, experiment=experiment)

        if self.verbose:
            print('Checking resources...')

        # Check if resource already exists
        if os.path.exists(skipgram_fp):
            params = self._dump_config(experiment=experiment)
            print(f'\nCONFLICT: Resource already exists for:\n')
            for k, v in params:
                print(f"{k:<26}{v}")
            print()
            raise
        
        # Check required resource exists
        assert os.path.exists(walk_fp), f'ERROR: Resource not found. Have you generated walks for for:\nBiogrid: {version}\np: {p}\nq: {q}\nNumber of walks: {num_walks}\nWalk length: {walk_length}'        

        # Process the skipgrams
        _ = self._make_uuid('skipgram', experiment=experiment, add_to_lookup=True) # Add the skipgram data asset to archive manager

        try: 

            skipgrams = TextDataset(walk_fp, skipgram_fp, verbose=self.verbose, negative_samples=negative_samples)
            skipgrams.create_skipgram_dataset(window_size=window_size, negative_samples=negative_samples)

            return skipgrams.data()
        
        except Exception as e:

            os.remove(skipgram_fp)
            print(f'ERROR: {e}')
            raise             
        

    def describe(self, experiment='current'):
        '''
        This function provides a description of the skipgram data assets.
        It describes the current config details, the structure of a trianing
        example, the size of the dataset and any features of the dataset.
        '''

        # Make the filepath and check it exists
        skipgram_fp = self._make_filepath('skipgram', experiment=experiment)

        assert os.path.exists(skipgram_fp), f'ERROR: resource not found:\n{skipgram_fp}'

        # Get config details
        version = self._get_biogrid_version(experiment=experiment)
        p, q = self._get_p_q_values(experiment=experiment)
        num_walks, walk_length = self._get_sampling_values(experiment=experiment)
        negative_samples, window_size = self._get_skipgram_values(experiment=experiment)     

        # Measure the size taken on disk
        size = round(os.stat(skipgram_fp).st_size / 1024**3, 6)   

        # Load the data
        walk_fp = ''
        skipgrams = TextDataset(walk_fp, skipgram_fp, negative_samples=negative_samples).data()

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


    def validate(self, experiment='current'):
        '''
        Things to validate
        > Intengrity of crc
        > Vocab size = num genes
        > Deserialised data is same shape
        '''

        # Get parameter values
        negative_samples, _ = self._get_skipgram_values(experiment=experiment)             

        # Make the filepath and check it exists
        walk_fp = self._make_filepath('walk', ext=True, experiment=experiment)
        skipgram_fp = self._make_filepath('skipgram', experiment=experiment)

        assert os.path.exists(walk_fp), f'ERROR: resource not found:\n{walk_fp}'
        assert os.path.exists(skipgram_fp), f'ERROR: resource not found:\n{skipgram_fp}'

        # Load the data
        skipgrams = TextDataset(walk_fp, skipgram_fp, negative_samples=negative_samples)        
        
        # CRC check
        if self.verbose:
            print('validating:\t', skipgram_fp)
        total_records, total_bad_len_crc, total_bad_data_crc = validate_dataset_slower([skipgram_fp], verbose=False)

        # Check vocab size
        vocab_size = len(skipgrams.text_vectorizer().get_vocabulary())
        walks = skipgrams._load_walks_as_iterator(walk_fp)
        unique_tokens = set()
        for walk in walks:
            data = walk.to_numpy()
            for (x, y), value in np.ndenumerate(data):
                if value not in unique_tokens:
                    unique_tokens.add(value)

        # Check tensor shape
        for example in skipgrams.data().take(1): # This is a batch
            target = example[0]['target']
            context = example[0]['context']
            label = example[1]

        ## Make the tables
        # Corruption
        corruption_table = PrettyTable()
        corruption_header = ['Total Records', 'Corrupted Length', 'Corrupted Data']
        corruption_table.field_names = corruption_header
        corruption_table.add_row([total_records, total_bad_len_crc, total_bad_data_crc])

        # Vocab size
        vocab_table = PrettyTable()
        vocab_header = ['Expected vocab size', 'Actual vocab size']
        vocab_table.field_names = vocab_header
        vocab_table.add_row([vocab_size, len(unique_tokens) + 1])      

        # Tensor shapes
        tensor_table = PrettyTable()
        tensor_header = ['Tensor', 'Expected Shape', 'Actual Shape']
        tensor_table.field_names = tensor_header
        tensor_table.add_row(['Target', (1,), target.shape])
        tensor_table.add_row(['Context', (negative_samples+1,), context.shape])         
        tensor_table.add_row(['Label', (negative_samples+1,), label.shape])           

        # Print tables
        if self.verbose:
            print(f'\nNumber of corrupted TFRecords:')
            print(corruption_table)
            print(f'\nVocab size:')
            print(vocab_table)
            print(f'\nTensor Sizes:')
            print(tensor_table)            
        

    def head(self, nrows=5, experiment='current'):
        '''
        Print 1 batch and the top n examples
        '''

        # Get parameter values
        negative_samples, _ = self._get_skipgram_values(experiment=experiment)     

        # Make the filepath and check it exists
        skipgram_fp = self._make_filepath('skipgram', experiment=experiment)

        assert os.path.exists(skipgram_fp), f'ERROR: resource not found:\n{skipgram_fp}'

        # Load the data
        walk_fp = ''
        skipgrams = TextDataset(walk_fp, skipgram_fp, negative_samples=negative_samples).data()        
        
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

                print(f'\nHead (n={nrows}):')
                for i in range(nrows):
                    t, c, l = next(data)
                    print('Target:\t\t', t)
                    print('Context:\t', c)
                    print('Label:\t\t', l, '\n')     

    

if __name__ == "__main__":
    skipgrams = Skipgrams()
    # skipgrams.process()
    # skipgrams.describe()
    # skipgrams.validate()
    skipgrams.head()