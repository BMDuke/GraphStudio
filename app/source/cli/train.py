#!/usr/bin/python

import sys
import yaml
import pprint

from prettytable import PrettyTable

from source.utils.config import Config
from source.utils.archive import Archive, ModelArchive

from source.datasets.text import TextDataset
from source.datasets.multi_label import MultiLabelDataset

from source.train.node2vec import Node2Vec
from source.train.multilabel_classifier import MultiLabelClassifier

'''

Documentation:  

'''

model_mapping = {
    'node2vec': [Node2Vec, TextDataset],
    'mlc': [MultiLabelClassifier, MultiLabelDataset],
    'bc': [],
}


if __name__ == "__main__":

    config = Config()

    pprinter = pprint.PrettyPrinter(indent=4)
    
    # Parse command line arguments
    command = sys.argv[1]  

    if command == 'options':

        pass

    elif command == 'train':

        try:
            arg =  sys.argv[2]    
        except Exception as e:
            out = f'\nNo argument passed to #. What model do you want to train? [model] | [model]=[experiment name] '
            raise ValueError(out)

        args = arg.split('=')
        if len(args) == 2:
            model, experiment = args
        else:
            model, experiment = args[0], 'default'

        model, dataset = model_mapping[model] # Just realised dataset not used

        model = model()

        model.train(experiment)



    elif command == 'ls':

        archive = ModelArchive()
        lookup = archive._load_lookup()

        table = PrettyTable()
        table.field_names = lookup.columns.values
        for i in range(lookup.index.size):
            table.add_row(lookup.iloc[ i ].values)

        print(table)


    elif command == 'rm':
        
        pass

    else:

        out = f'\nNo command matching {command} found. Please ensure you have entered the correct command, or refer to docs for help.'
        raise ValueError(out)




    

    