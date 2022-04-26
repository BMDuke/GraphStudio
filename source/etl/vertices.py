import os
import random
import shutil
import tqdm
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from prettytable import PrettyTable
import tensorflow_data_validation as tfdv
import numpy as np

from source.utils.config import Config
from source.utils.archive import Archive
from source.etl.msig import MSig
from source.datasets.multi_label import MultiLabelDataset

class Vertices(object):
    
    '''
    This class is the ETL utility to generate the training data for the multi-
    label classification network. It uses the MultiLabelDataset class to 
    handle the data transformationa and the Archive and Config classes
    to handle data asset management. 

    About:
    The input into this class are labelled gene_id pairs. This is a list of 
    [[label, gene], [label, gene], ... ] pairs which are read in from files in
    the data/processed/gene_ids dir. This is the location of the Msig class 
    output. These pairs are then grouped by gene, and then the corresponing
    list of labels are encoded into a multi-hot encoded vector. This is 
    format which will be used as input to the multilabel classifier. 

    Public API:
     - process()                Generate the multi-hot encodings and save to disk
     - describe()               Describe the dataset produced
     - validate()               Validate the dataset
     - head()                   Print a sample of the data

    '''

    # Class attrs
    base_dir = 'data/processed'
    out_file = 'multilabelled_genes.tfrecord'


    def __init__(self, config, debug=False, verbose=True):
        '''
        Instantiate a new instance of the Vertices ETL tool. It takes a config 
        object as an argument which allows it to retrieve current project 
        parameter values.
        '''
        self.config = config
        self.debug = debug
        self.verbose = verbose


    def process(self):
        '''
        This function reads in the labelled gene ids, which are output by the MSig class
        and transforms them into a tf Dataset where x is the gene id and y is a 
        multi-hot encoding. 
        '''

        if self.verbose:
            print('Creating multi-hot encoded dataset for multilabel classification...')

        # Create file paths
        msig = MSig(self.config)
        msig_fp = msig._get_destination()
        vertices_fp = os.path.join(self.base_dir, 'vertices', self.out_file)
        
        # Check required resource exists and target has not already 
        # been processed
        assert os.path.exists(msig_fp), f'ERROR: Resource not found {msig_fp} - Have you processed the msig dataset?'
        assert not os.path.exists(vertices_fp), f'ERROR: Resource conflict {vertices_fp} already exists'        
        
        # Create multilabelled dataset 
        pairs = msig._load_msig()
        multilabel = MultiLabelDataset()
        data = multilabel.create_dataset(pairs=pairs.values)

        # Save data
        multilabel.save_dataset(data, vertices_fp)

        if self.verbose:
            print('Creating multi-hot encoded dataset for multilabel classification...')        

        return data
        


    def describe(self):
        '''
        This fucntions provides a description and an example of the data asset that we
        are creating. The number of genes, labels annd number of labels per gene should
        be the same as MSig.describe().
        '''

        # Check resource exists
        vertices_fp = os.path.join(self.base_dir, 'vertices', self.out_file)

        assert os.path.exists(vertices_fp), f'ERROR: Resource not found: {vertices_fp} \nHave you processed the multilabelled data?'        

        # Load the data
        multilabel = MultiLabelDataset(filepath=vertices_fp)
        data = multilabel.data()

        # Number of genes
        num_genes = len(list(data.unbatch().as_numpy_iterator()))

        # Number of labels
        for example in data.unbatch().take(1):
            print(example[1])
            labels = example[1]
            number_of_labels = len(labels)

        # Number of labels per gene
        counts = {gene.numpy(): np.sum(labels.numpy()) for gene, labels in data.unbatch()}
        labels_per_gene = {}
        for gene, num_labels in counts.items():
            if num_labels in labels_per_gene.keys():
                labels_per_gene[num_labels] += 1
            else:
                labels_per_gene[num_labels] = 1

        # Disk utilisation
        size = round(os.stat(vertices_fp).st_size / 1024**3, 6)   

        # Get one example
        for i in data.unbatch().take(1): # This is a batch
            x = i[0]
            y = i[1]
            
        ## Make the tables
        # Data
        data_table = PrettyTable()
        data_header = ['Number of genes', 'Number of labels']
        data_table.field_names = data_header
        data_table.add_row([num_genes, number_of_labels])

        # Labels per gene
        labels_table = PrettyTable()
        labels_header = ['Number classes per gene', 'Number of genes']
        labels_table.field_names = labels_header
        labels_table.add_rows([[k, v] for k, v in labels_per_gene.items()])

        # Disk spce
        disk_table = PrettyTable()
        disk_header = ['Disk - GB']
        disk_table.field_names = disk_header
        disk_table.add_row([size])     

        # Example descirption
        example_table = PrettyTable()
        example_header = ['Variable', 'Value']
        example_table.field_names = example_header
        example_table.add_row(['x', x])
        example_table.add_row(['y', y])

        ## Print Tables
        if self.verbose:
            print(f'\nExample:')
            print(example_table)              
            print(f'\nData Details:')
            print(data_table)
            print(labels_table)
            print(f'\nDisk Utilisation Details:')
            print(disk_table)
          
        



    def validate(self):
        '''
        '''
        # Check resource exists        
        # Schema validation

        

    def head(self, n=5):
        '''
        Print the first n examples from the dataset
        '''

        # Check resource exists
        vertices_fp = os.path.join(self.base_dir, 'vertices', self.out_file)

        assert os.path.exists(vertices_fp), f'ERROR: Resource not found: {vertices_fp} \nHave you processed the multilabelled data?'        

        # Load the data
        multilabel = MultiLabelDataset(filepath=vertices_fp)
        data = multilabel.data()

        # Print the first n examples
        if self.verbose:
            for x, y in data.unbatch().take(n):
                print(f'\nx: {x}')
                print(f'y: {y}')        

        
               

    

if __name__ == "__main__":
    conf = Config()
    vertices = Vertices(conf)
    # data = vertices.process()
    # vertices.describe()
    vertices.head()
    # print(data)