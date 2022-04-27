#!/usr/bin/python

'''

Documentation:  

This script defines the CLI for managing project config files

Accepted Kwargs: 

    -v,         This is the version of BioGrid to be used.
    --version   If multiple instances of this flag are passed
                then the first one will be taken. Versions have
                the format X.Y.ABC. More information can be found
                https://downloads.thebiogrid.org/BioGRID/Release-Archive/

    -e,         This is the email address you have used to register your 
    --email     account for gene enrichment dataset. To register, see here
                http://www.gsea-msigdb.org/gsea/login.jsp. For more 
                information, see here
                http://www.gsea-msigdb.org/gsea/index.jsp


Things to do:
>  



'''

import sys
import yaml

from source.utils.config import Config

def parse_directive(args):
    '''
    Predicate checking for 'experiment' directive in command line args
    args            sys.argv obejct (list)
    '''
    return args[2] == 'experiment'


if __name__ == "__main__":

    config = Config()

    # Parse command line arguments
    command = sys.argv[1]

    # Handle new commands
    if command == "new":

        is_experiment = parse_directive(sys.argv)

        # Create new model config
        if is_experiment:
            model = sys.argv[3]
            name = sys.argv[4]
            params = sys.argv[5:]

            config.add_experiment(model, name, *params)

        # Create new config file
        else:
            name = sys.argv[2]
            try:
                params = sys.argv[3:]
            except: 
                params = None           
            

            config.new(name, *params)

    # Handle use commands
    elif command == "use":
        
        is_experiment = parse_directive(sys.argv)

        # Set model config 
        if is_experiment:
            model = sys.argv[3]
            name = sys.argv[4]

            config.set_experiment(model, name)            
        
        # Set config file
        else:
            name = sys.argv[2]

            config.set(name)

    # Handle show commands
    elif command == "show":
        
        try:

            name = sys.argv[2]
        
        except: 

            name = None

        conf = config.show(name)
        print(yaml.dump(conf))

    # Handle current commands
    elif command == "current":

        conf = config.current()    
        print(conf)

    # Handle edit commands
    elif command == "edit":
        
        is_experiment = parse_directive(sys.argv)

        # Make changes to a model config
        if is_experiment:
            model = sys.argv[3]
            name = sys.argv[4]
            params = sys.argv[5:]

            config.edit_experiment(model, name, *params)

        # Make changes to the data config
        else:
            name = sys.argv[2]
            params = sys.argv[3:]

            config.edit(name, *params)                        

    # Handle delete commands
    elif command == "delete":
        
        is_experiment = parse_directive(sys.argv)

        # Delete a model config
        if is_experiment:
            model = sys.argv[3]
            name = sys.argv[4]

            config.delete_experiment(model, name)

        # Delete a config file
        else: 
            try:
                name = sys.argv[2]
            except: 
                name = None

            config.delete(name)
    
    else:

        raise ValueError(f"ERROR: argument'{command}' unknown")