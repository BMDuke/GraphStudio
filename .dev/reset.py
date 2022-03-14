#!/usr/bin/python

'''

Documentation:  

This file is to be used to reset the project structure 
during development. 

Given the path of a file, this script will delete the
file. Given the path of a directory, this file will 
delete all the files and subdirectories in that 
directory. 

To use this script, run from the commandline and pass 
filepaths that you would like to have removed as 
kwargs. 

    -p,         Relative path to file or directory to be reset.
    --path      This flag can be passed multiple times.



Things to do:
> Protect original project structure from erasure
> Create a data structure that represents the the project
    structure. Use this to support deleting of directories
    high up on the file tree. 

'''

import sys
import os 
import shutil


if __name__ == "__main__":

    working_dir = os.getcwd()

    # Collect paths to be reset (-p or --path)
    targets = [arg.split('=')[1] for arg in sys.argv 
                if (arg.split('=')[0] == '-p') or (arg.split('=')[0] == '--path')]

    for target in targets:

        filepath = os.path.join(working_dir, target)
        
        try: 

            isFile = os.path.isfile(filepath)
            isLink = os.path.islink(filepath)
            isDir = os.path.isdir(filepath)

            if isFile or isLink:
                os.unlink(filepath)
                print(f"REMOVED: {target}")
            elif isDir:
                shutil.rmtree(filepath)
                os.mkdir(filepath)
                print(f"REMOVED: {target}")

        except Exception as e:

            print(f"Failed to delete {filepath} \nReason {e}")