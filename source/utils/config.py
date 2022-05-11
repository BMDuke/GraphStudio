import os
import re
import ast
import shutil
import pathlib
from tokenize import Name

import yaml
import pprint

filepath = "conf"
current = "__current__.yaml"
template = "__template__.yaml"

class Config(object):

    '''

    This is a class to handle the maintenance and management of 
    config files to be used in the project. 

    Things to do: 
    > Remodel the instance variables fp, current_fp and template_fp to 
        class variables
    > Give the config a static method to return the current
        config
    > Provide commentary via stdout
    > This should be modelled as a singleton class

    '''    

    def __init__(self):
        self.fp = filepath
        self.current_fp = current
        self.template_fp = template        

    def new(self, name, *params):
        '''
        Create new config
        
        UPDATE: 
         - *params now only expects to be the biogrid dataset
            version because the data category now handles experiments
            like the models
        '''

        # Create file path
        # dest = os.path.join(self.fp, f"{name}.yaml")       

        # Copy the file
        # _ = shutil.copyfile(self.template_fp, dest)

        # Edit the file

        # Create the filepath based on the name provided
        # Check if the file already exists -if yes error
        # Duplicate the template file to the destination file
        # Edit the destination file
        config = self._make_filepath(name)

        if config.exists():
            raise NameError(f"Config {name}.yaml already exists")
        
        default_conf = self._new_default_config()

        self._save(name, default_conf)

        self.edit(name, *params)

        self.set(name) # Switch to that config context


    def set(self, name):
        '''
        Set the active config file for the project
        '''
        
        assert name is not None, "ERROR: Config.set() - 'name' not provided. Which config file do you want to set?"

        config = self._make_filepath(name)

        if config.exists():
            current = self._load(self.current_fp)
            current["current"] = name
            _ = self._save(self.current_fp, current)
        else:
            raise NameError(f"Config '{name}.yaml' does not exists")

    def show(self, name=None):
        '''
        Print the config file
        '''

        # If name is not supplied:
            # Load the current file and extract current config 
            # Load the current config
            # name = current config
        # Print the named config to the terminal
        # Return the named config 
        # ** this should return the object, it should be the job of 
        # run/config.py to print it. 

        if not name:
            current = self._load(self.current_fp)
            name = current["current"]
        
        file = self._make_filepath(name)

        if file.exists():
            
            config = self._load(name)
        
        else:

            raise NameError(f"ERROR: Config \'{name}\' not found")

        return config

    def current(self):
        '''
        Print the current config file used for the project
        '''
        pass
        
        # Load the current file and extract the current config
        # Print the current config
        # return current
        current = self._load(self.current_fp)
        name = current["current"]

        return name        

    def edit(self, name, *params):
        '''
        Edit the config file
        '''

        # Check the supplied file exists
        # Load the file 
        # Apply the updates to the config
        # Save the updated version
         
                
        file = self._make_filepath(name)

        if file.exists():
            config = self._load(name)
            config['data'] = self._update_values(config['data'], *params)
            _ = self._save(name, config)
        else:
            raise NameError(f"Config '{name}.yaml' does not exists")


    def delete(self, name, force=False):
        '''
        Delete the config file
        '''

        # Check the file exists
        # Prompt confirmation from user
        # Delete the file

        config = self._make_filepath(name)

        if config.exists():
            if not force:
                confirmation = input(f"ATTENTION: Are you sure you want to delete {name}.yaml? [ y / n ]")
                if confirmation.lower() == 'y':
                    os.remove(config)
            else:
                os.remove(config)
        else:
            raise NameError(f"Config '{name}.yaml' does not exists")        
        pass             

    def add_experiment(self, model, name, *params):
        '''
        Add a model configuration to the config file

        ** This will only add experiments to the current
            config file
        '''

        # Check name argument is supplied
        # If name argumant is not supplied
        #   Load current and extract current config
        #   name = current config
        # Make blank template section for model
        # Apply user given values to template section
        # Load current config 
        # Append complete template to model experiments
        # Save the config 

        current = self.current() # name of the current config
        current_conf = self._load(current) # the current config

        default_conf = self._new_default_config(model)
        model_conf = self._update_values(default_conf, *params)

        current_conf[model]["experiments"][name] = model_conf

        _ = self._save(current, current_conf)


    def set_experiment(self, model, name):
        '''
        Set the active config for a given model

        ** This will only set experiments in the current
            config file
        '''

        # Check name argument is supplied 
        # If name argumant is not supplied
        #   Load current and extract current config
        #   name = current config
        # Load named config
        # Check experiment is in model
        # Set model current to experiment

        current = self.current() # name of the current config
        current_conf = self._load(current) # the current config

        assert name in current_conf[model]["experiments"], f"ERROR: experiment '{name}' not found for {model}"

        current_conf[model]["current"] = name

        _ = self._save(current, current_conf)

    def edit_experiment(self, model, name, *params):
        '''
        Edit the config for a given model

        ** This will only edit experiments in the current
            config file        
        '''

        # Check name argument is supplied 
        # If name argumant is not supplied
        #   Load current and extract current config
        #   name = current config   
        # Check experiment is in model
        # Select experiment from model
        # Apply values to experiment
        # Save config

        current = self.current() # name of the current config
        current_conf = self._load(current) # the current config        

        assert name in current_conf[model]["experiments"], f"ERROR: experiment '{name}' not found for {model}"

        model_conf = current_conf[model]["experiments"][name]
        model_conf = self._update_values(model_conf, *params)
        current_conf[model]["experiments"][name] = model_conf

        _ = self._save(current, current_conf)

    def delete_experiment(self, model, name):
        '''
        Delete the config for a given model
        '''

        # Check name argument is supplied 
        # If name argumant is not supplied
        #   Load current and extract current config
        #   name = current config
        # Check experiment is in model
        # remove experiment from model 
        # Save model

        current = self.current() # name of the current config
        current_conf = self._load(current) # the current config        

        assert name in current_conf[model]["experiments"], f"ERROR: experiment '{name}' not found for {model}"

        del current_conf[model]["experiments"][name]

        _ = self._save(current, current_conf)


    def get_experiment(self, asset, name='current'):
        '''
        This gets the config values for a given experiemnt.
        asset:          This is the project asset requested
                        data, node2vec, multi_label etc...
        name:           This is the name of the experiment, 
                        the default value is current
        '''
        current = self.current() # name of the current config
        current_conf = self._load(current) # the current config          

        # Define a mapping between project asset and relevant config parameters
        mapping = {
            'data': ['data:version'],
            'skipgram': ['data:version', f'data:{name}'],
            'node2vec': [f'node2vec:{name}'],
            'binary_classifier': [f'binaryClassifier:{name}'],
            'multi_label': [f'multiLabelClassifier:{name}']
        }  

        assert asset in mapping.keys(), f'ERROR: item: {asset} not found in config keys'

        # Get keys to return 
        keys = mapping[asset]

        # Loop through and get values
        config = {}
        for key in keys:
            resource, parameter = key.split(':')
            if parameter == 'version':
                config[parameter] = current_conf[resource][parameter]
            elif parameter == 'current':
                current = current_conf[resource][parameter]
                values =  current_conf[resource]['experiments'][current]
                for k, v in values.items():
                    config[k] = v
            else:
                values =  current_conf[resource]['experiments'][parameter]
                for k, v in values.items():
                    config[k] = v
        
        return config              

   
    def _load(self, name):
        
        filepath = self._make_filepath(name)

        # load the yaml file or error
        try: 
            
            with open(filepath, 'r') as f:
                conf = yaml.load(f, Loader=yaml.FullLoader)
                return conf

        except Exception as e:
            print(f"ERROR: Config._load {e}")

    def _save(self, name, data):

        filepath = self._make_filepath(name)

        # save the yaml file or error
        try: 
            
            with open(filepath, 'w') as f:
                data = yaml.dump(data, f)
                return data

        except Exception as e:
            print(f"ERROR: Config._save {e}")            


    def _update_values(self, conf, *params):

        # convert 'param=value' to {param:value}
        _params = {p.split('=')[0]: p.split('=')[1] for p in params}

        for k, v in _params.items():
            if k in conf:
                conf[k] = self._guess_type(v)
            else:
                raise KeyError(f"ERROR: Key {k} not found in config object {conf}")
        
        return conf

    def _make_filepath(self, name):
        
        # make the filepath from the name 
        file = name if name.endswith('.yaml') else f"{name}.yaml"
        filepath = os.path.join(self.fp, file)

        return pathlib.Path(filepath)

    def _new_default_config(self, section=None):
        
        # Load config template 
        # If section exists return that section
        # Else return the config

        default_conf = self._load(self.template_fp)

        if section:
            return default_conf[section]["experiments"]["default"]
        
        return default_conf

    def _guess_type(self, inp):
        
        '''
        Guess the type of a string. 
        If too hard, return string
        '''

        # pattern for version
        pattern = re.compile("\d\.\d\.\d{3}")
        match = pattern.match(inp)

        if match is not None:
            return inp       # return as str
        
        else:

            try:
                return ast.literal_eval(inp)    # convert to numeric type    
            except ValueError:
                return inp




