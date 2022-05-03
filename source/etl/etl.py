import os
import random
import shutil
import tqdm
from pathlib import Path

from source.utils.config import Config
from source.utils.archive import Archive

class ETL(object):

    '''
    This is a base class for other ETL classes, 
    providing shared methods for managing config 
    and archive files. 
    
    This was written as part of a refactor after
    the initial ETL classes had been written.
    
    For this reason, there may exist duplicated code
    in the code base until everything has been tidied 
    up. 
    
    The main contribution of this class is that it:
    > Automatically instantiates a config object
    > Automatically instantiates an archive object
    > Provides shared methods for accessing config
        values
    > Provides additional functionality to be able to 
        return configs of specified experiments rather
        than only the default

    '''

    # Class attrs
    base_dir = 'data/processed'
    graph_format = 'pickle'


    def __init__(self):
        '''
        Initialise a config and an archive object
        '''

        config = Config()
        self.config = config
        self.data_archive = Archive(config)

    def _get_biogrid_version(self):
        '''
        This returns the version of the biogrid dataset
        specified in the config file.
        '''

        return self.config.get_experiment('data').get('version')

    def _get_p_q_values(self, experiment='current'):
        '''
        This function returns the values of p and q that 
        have been specified in a given experiment. 
        
        If no value for experiment is provided then config 
        returns the values of the current experiment
        '''
        
        data_conf = self.config.get_experiment('skipgram', name=experiment)

        return data_conf.get('p'), data_conf.get('q')

    def _get_sampling_values(self, experiment='current'):
        '''
        This function returns the values of num_walks and 
        walk_length specified from a gieven experiment.
        
        If no value for experiment is provided then config 
        returns the values of the current experiment
        '''
        
        data_conf = self.config.get_experiment('skipgram', name=experiment)

        return data_conf.get('num_walks'), data_conf.get('walk_length')

    def _get_skipgram_values(self, experiment='current'):
        '''
        This function returns the values of num_walks and 
        walk_length specified from a gieven experiment.
        
        If no value for experiment is provided then config 
        returns the values of the current experiment
        '''
        
        data_conf = self.config.get_experiment('skipgram', name=experiment)

        return data_conf.get('negative_samples'), data_conf.get('window_size')   

    def _dump_config(self, experiment='current'):
        '''
        This function return key value pairs for a given experiment. 
        If no expeirment is provided, then the current values are returned.
        '''

        data_conf = self.config.get_experiment('skipgram', name=experiment)

        dump = [[k, v] for k, v in data_conf.items()]

        return dump    

    def _make_uuid(self, resource, experiment='current', add_to_lookup=False):
        '''
        This function generates a unique id for a given 
        resource, based on a combination of parameters. 
        '''

        data_conf = self.config.get_experiment('skipgram', name=experiment)

        uuid = self.data_archive.make_id_from_dict(resource, 
                                                    data_conf, 
                                                    add_to_lookup=add_to_lookup)
        
        return uuid



    def _make_filepath(self, resource, experiment='current', ext=True):
        '''
        This makes returns the filepath for the requested resource.
        resource:           (str) biogrid, transition or walk
        experiment:         (str) Data experiment to use
        ext:                (bool) Should extension be added
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
            },
            'skipgram': {
                'dir': 'skipgrams',
                'extension': 'tfrecord'
            }            
        }

        # Check exists
        assert resource in mapping.keys(), f'ERROR: Requested resource not found {resource}. Choose from [{str(*list(mapping.keys()))}].'

        # Make path to directory 
        path = self.base_dir
        path = os.path.join(path, mapping[resource]['dir'])

        # Generate uuid
        uuid = self._make_uuid(resource, experiment=experiment)

        # Make filename
        filename = uuid
        if ext:
            filename = f'{uuid}.{mapping[resource]["extension"]}'

        return os.path.join(path, filename)        



if __name__ == "__main__":
    etl = ETL()
    print(etl._get_biogrid_version())
    print(etl._get_p_q_values())
    print(etl._get_sampling_values())
    print(etl._get_skipgram_values())
    print(etl._dump_config())
    print(etl._make_uuid('biogrid'))
    print(etl._make_uuid('transition'))
    print(etl._make_uuid('walk'))
    print(etl._make_uuid('skipgram'))
    print(etl._make_filepath('biogrid'))
    print(etl._make_filepath('transition'))
    print(etl._make_filepath('walk'))
    print(etl._make_filepath('skipgram'))