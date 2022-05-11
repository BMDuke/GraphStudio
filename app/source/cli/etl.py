#!/usr/bin/python

import sys
import yaml
import pprint

from prettytable import PrettyTable

from source.utils.config import Config
from source.utils.archive import Archive

from source.etl.msig import MSig
from source.etl.biogrid import BioGrid
from source.etl.transition_probs import TransitionProb
from source.etl.walks import Walk
from source.etl.skipgrams import Skipgrams
from source.etl.vertices import Vertices

'''

Documentation:  

'''

dataset_mapping = {
    'msig': MSig,
    'biogrid': BioGrid,
    'transition_probs': TransitionProb,
    'walks': Walk,
    'skipgrams': Skipgrams,
    'vertices': Vertices
}


if __name__ == "__main__":

    config = Config()

    pprinter = pprint.PrettyPrinter(indent=4)
    
    # Parse command line arguments
    command = sys.argv[1]  

    if command == 'process':

        try:
            arg =  sys.argv[2]    
        except Exception as e:
            out = f'\nNo argument passed to process. What dataset do you want to process? \'all\' | [dataset] | [dataset]=[experiment name] '
            raise ValueError(out)

        if arg == 'all':

            # print('ARG IS ALL')

            args = arg.split('=')
            if len(args) == 2:
                _, experiment = args
            else:
                _, experiment = args[0], 'current'

            msig = MSig(config)
            biogrid = BioGrid(config)
            transition_probs = TransitionProb(config)
            walks = Walk(config)
            skipgrams = Skipgrams(config)
            vertices = Vertices(config)

            msig.process()
            biogrid.process()
            transition_probs.process(experiment=experiment)
            walks.process(experiment=experiment)
            skipgrams.process(experiment=experiment)
            vertices.process()

        elif arg:

            # print('ARG IS DATASET ', arg)
            args = arg.split('=')
            if len(args) == 2:
                dataset, experiment = args
            else:
                dataset, experiment = args[0], 'current'

            assert dataset in dataset_mapping.keys(), f'ERROR dataset \'{dataset}\' not recognised, please choose from {str(list(dataset_mapping.keys()))}'

            data = dataset_mapping[dataset]
            data = data()

            data.process(experiment=experiment)

        else:
            raise 

    elif command == 'describe':

        try:
            arg =  sys.argv[2]    
        except Exception as e:
            out = f'\nNo argument passed to describe. What dataset do you want to decribe? [dataset] | [dataset]=[experiment name] '
            raise ValueError(out)

        # print('ARG IS DATASET ', arg)
        args = arg.split('=')
        if len(args) == 2:
            dataset, experiment = args
        else:
            dataset, experiment = args[0], 'current'

        assert dataset in dataset_mapping.keys(), f'ERROR dataset \'{dataset}\' not recognised, please choose from {str(list(dataset_mapping.keys()))}'

        data = dataset_mapping[dataset]
        data = data()

        data.describe(experiment=experiment)

    elif command == 'validate':
        
        try:
            arg =  sys.argv[2]    
        except Exception as e:
            out = f'\nNo argument passed to validate. What dataset do you want to validate? [dataset] | [dataset]=[experiment name] '
            raise ValueError(out)

        # print('ARG IS DATASET ', arg)
        args = arg.split('=')
        if len(args) == 2:
            dataset, experiment = args
        else:
            dataset, experiment = args[0], 'current'
        
        assert dataset in dataset_mapping.keys(), f'ERROR dataset \'{dataset}\' not recognised, please choose from {str(list(dataset_mapping.keys()))}'

        data = dataset_mapping[dataset]
        data = data()

        data.validate(experiment=experiment)


    elif command == 'head':
        
        try:
            arg =  sys.argv[2]    
        except Exception as e:
            out = f'\nNo argument passed to head. What dataset do you want to view? [dataset] | [dataset]=[experiment name] '
            raise ValueError(out)

        # print('ARG IS DATASET ', arg)
        args = arg.split('=')
        if len(args) == 2:
            dataset, experiment = args
        else:
            dataset, experiment = args[0], 'current'

        assert dataset in dataset_mapping.keys(), f'ERROR dataset \'{dataset}\' not recognised, please choose from {str(list(dataset_mapping.keys()))}'

        data = dataset_mapping[dataset]
        data = data()

        nrows=5
        try:
            arg =  sys.argv[3]   
            nrows = int(arg.split('=')[1])
            print(nrows)
        except Exception as e:
            pass    

        data.head(experiment=experiment, nrows=nrows)


    elif command == 'ls':

        archive = Archive(Config())
        lookup = archive._load_lookup()


        table = PrettyTable()
        table.field_names = lookup.columns.values
        for i in range(lookup.index.stop):
            table.add_row(lookup.iloc[ i ].values)

        print(table)
        

    elif command == 'rm':

        try:
            arg =  sys.argv[2]    
        except Exception as e:
            out = f'\nNo argument passed to rm. What dataset do you want to delete? [dataset] | [dataset]=[experiment name] '
            raise ValueError(out)

        # print('ARG IS DATASET ', arg)
        args = arg.split('=')
        if len(args) == 2:
            dataset, experiment = args
        else:
            dataset, experiment = args[0], None
        print(dataset, experiment)

    else:

        out = f'\nNo command matching {command} found. Please ensure you have entered the correct command, or refer to docs for help.'
        raise ValueError(out)



# # Statement templates for ETL CLI
# 'cmd:process arg:all arg:<dataset> kwarg:<dataset>:<experiment> arglen:1'
# 'cmd:describe arg:<dataset> kwarg:<dataset>:<experiment> arglen:1'
# 'cmd:head arg:<dataset> kwarg:<dataset>:<experiment> okwarg:n:<nrows> argrange:1:2'
# 'cmd:validate arg:<dataset> kwarg:<dataset>:<experiment> arglen:1'
# 'cmd:ls arglen:0'
# 'cmd:rm arg:<dataset> kwarg:<dataset>:<experiment> arglen:1'

# # Statement templates for train CLI
# 'cmd:options arglen:0'
# 'arg:<dataset> kwarg:<dataset>:<experiment> arglen:1'
# 'cmd:ls arglen:0'
# 'cmd:process arg:all arg:<dataset> kwarg:<dataset>:<experiment> arglen:1'

# # Statement templates for config CLI
# 'cmd:new qual:experiment kwarg:<asset>:<name> qual:config arg:<name> arglen:1'
# 'cmd:use qual:experiment kwarg:<asset>:<name> qual:config arg:<name> arglen:1'
# 'cmd:show qual:experiment kwarg:<asset>:<name> qual:config arg:<name> arglen:1'
# 'cmd:current qual:experiment arg:<name> arglen:1 qual:config arglen:0'
# 'cmd:edit qual:experiment kwarg:<asset>:<name> vkwarg:<param>:<value> arglen:-1 qual:config arg:<name> arglen:1'

# '''
#     cmd:process 
#         arg:all                         # literal 
#         arg:<dataset>                   # variable
#         kwarg:<dataset>:<experiment>    # key, value pair
#         arglen:1                        # literal
# '''

# '''
#     cmd:edit 
#         qual:experiment 
#             kwarg:<asset>:<name> 
#             vkwarg:<param>:<value> 
#             arglen:-1 
#         qual:config 
#             arg:<name> 
#             arglen:1
# '''

# rule = {
#     'command': 'process',
#     'qualifiers': {
#         None: {
#             'args': ['all', '<dataset>'],
#             'kwargs': ['<dataset>:<experiment>'],
#             'vkwargs': [],
#             'okwargs': [],
#             'min_args': 1,
#             'max_args': 1
#         } 
#     }
# }

# rule = {
#     'command': 'edit',
#     'qualifiers': {
#         'experiment': {
#             'args': [],
#             'kwargs': ['<asset>:<name>'],
#             'vkwargs': ['<param>:<value>'],
#             'okwargs': [],
#             'min_args': 1,
#             'max_args': 1
#         },
#         'config': {
#             'args': ['<name>'],
#             'kwargs': [],
#             'vkwargs': [],
#             'okwargs': [],
#             'min_args': 1,
#             'max_args': 1
#         }         
#     }
# }

# {
#     'command': 'edit',
#     'qualifier': 'experiment',
#     'args': [],
#     'kwargs': {'asset':'data', 'name':'default'},
#     'vkwargs':[{'param':'p', 'value':'0.5'}, {'param':'q', 'value':'0.7'}],
#     'okwargs': [],
# }


# rule = 'cmd:new qual:experiment kwarg:<asset>:<name> qual:config arg:<name> arglen:1'

# template = {
#     'command': None,
#     'qualifiers': {
#         None: {
#             'args': [],
#             'kwargs': [],
#             'vkwargs': [],
#             'okwargs': [],
#             'min_args': 0,
#             'max_args': 0
#         } 
#     }
# }

# def new_template():
#     return {
#         'command': None,
#         'qualifiers': {
#             None: {
#                 'args': [],
#                 'kwargs': [],
#                 'vkwargs': [],
#                 'okwargs': [],
#                 'min_args': 0,
#                 'max_args': 0
#             } 
#         }
#     }

# def parse_rule(rule):
#     '''
#     '''
#     tokens = rule.split(' ')

#     r = new_template()
#     qualifier = None 

#     for token in tokens:
        
#         commands = token.split(':')
        
#         command = commands.pop(0)

#         if command == 'cmd':
            
#             assert len(commands) == 1
#             r['command'] = commands.pop(0)

#         elif command == 'qual':
            
#             assert len(commands) == 1
#             q_old = qualifier
#             qualifier = commands.pop(0)
#             r['qualifiers'][qualifier] = new_template()['qualifiers'].get(None)
#             # print(r)
#             # print(new_template()['qualifiers'].get(None))

#         elif command == 'arg':

#             assert len(commands) == 1
#             if not r['qualifiers'][qualifier]['args']:
#                 r['qualifiers'][qualifier]['args'] = [commands.pop(0)]
#             else:
#                 r['qualifiers'][qualifier]['args'].append(commands.pop(0))
        
#         elif command == 'kwarg':

#             assert len(commands) == 2
#             if not r['qualifiers'][qualifier]['kwargs']:
#                 r['qualifiers'][qualifier]['kwargs'] = [':'.join(commands)]
#             else:
#                 r['qualifiers'][qualifier]['kwargs'].append(':'.join(commands))

#         elif command == 'vkwarg':

#             assert len(commands) == 2
#             if not r['qualifiers'][qualifier]['vkwargs']:
#                 r['qualifiers'][qualifier]['vkwargs'] = [':'.join(commands)]
#             else:
#                 r['qualifiers'][qualifier]['vkwargs'].append(':'.join(commands))

#         elif command == 'okwarg':

#             assert len(commands) == 2
#             if not r['qualifiers'][qualifier]['okwargs']:
#                 r['qualifiers'][qualifier]['okwargs'] = [':'.join(commands)]
#             else:
#                 r['qualifiers'][qualifier]['okwargs'].append(':'.join(commands))       

#         elif command == 'arglen':

#             assert len(commands) <= 2

#             if len(commands) == 1:
                
#                 length = commands.pop(0)
#                 r['qualifiers'][qualifier]['min_args'] = length
#                 r['qualifiers'][qualifier]['max_args'] = length

#             elif len(commands) == 2:
                
#                 r['qualifiers'][qualifier]['min_args'] = commands.pop(0)
#                 r['qualifiers'][qualifier]['max_args'] = commands.pop(0)       


#         else:
            
#             "Key error"
#             pass
    
#     return r

# # kwargs have preference over args
# # arglen is min and max argument length
# # minlen is min argument length
# # maxlen is max argument length
# # argrange is min to max number of args
# # A statement set can support 1 default action, Ie train train <model> is redundat
# # okwarg is an optional kwarg key must be a literal
# # vkwarg is a list of kwargs or a variable number of kwargs
# # Arglen -1 means variable     

    

    