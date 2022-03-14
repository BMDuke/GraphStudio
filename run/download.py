#!/usr/bin/python

'''

Documentation:  

This script is responsible for downloading the human PPI
dataset, extracting the relevant information, and transforming
it into a format suitable as an input into our model 
(a networkx graph). 

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
>  Handle authentication failure for MSigDB



'''

import os
import sys
import re
import shutil
import requests
import urllib 
from zipfile import ZipFile



RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

BIOGRID_V = '4.4.207'

MSIG_FILENAME = "gene_set.entrez.gmt"

if __name__ == "__main__":

    working_dir = os.getcwd()

    # 1. Extract command line arguments --<flag>=<arg> or -<f>=<arg>
    version = [arg.split('=')[1] for arg in sys.argv
                if (arg.split('=')[0] == '-v') or (arg.split('=')[0] == '--version')]

    email = [arg.split('=')[1] for arg in sys.argv
                if (arg.split('=')[0] == '-e') or (arg.split('=')[0] == '--email')]

    if version: 

        pattern = re.compile("\d\.\d\.\d{3}")
        match = pattern.match(version[0])
        if match is not None:
            version = match.group()
            print(f"VERSION: {version}")
        else:
            error = f"\nERROR: Pattern x.x.xxx not matched for version {version[0]} \nPlease see versions listed at https://downloads.thebiogrid.org/BioGRID/Release-Archive/"
            raise ValueError(error)
    
    else:

        version = BIOGRID_V

    if not email:

        error = "ERROR: Please provide a valid email address"
        raise NameError(error)      

    else:

        email = email[0]  

    # 2. Fetch the BioGrid data
    url = f"https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{version}/BIOGRID-ALL-{version}.tab3.zip"
    filepath = os.path.join(working_dir, RAW_DIR)
    zipped_filepath = os.path.join(working_dir, RAW_DIR, f"{version}.zip")
    unzipped_filepath = os.path.join(working_dir, RAW_DIR, f"{version}.txt")    

    zipped_version_exists = os.path.exists(zipped_filepath)
    unzipped_version_exists = os.path.exists(unzipped_filepath)

    if unzipped_version_exists: 

        print(f"File {version}.txt already exists in {filepath} \nSkipping download...")

    else:

        if not zipped_version_exists: 

            print("Fetching biogrid data... ")
            try:

                _, header = urllib.request.urlretrieve(url, zipped_filepath)

            except Exception as e:

                print(f"DOWNLOAD UNSUCCESSFUL {version} \nReason {e}")

            print(header)
            print(f"SUCCESS: data downloaded successfully\n{zipped_filepath}")
        
        else:

            print(f"File {version}.zip already exists in {filepath} \nSkipping download...")

        # 3. Unzip the data
        print(f"Unzipping {version}.zip ...")
        try: 
            
            dir = os.path.splitext(unzipped_filepath)[0] # extract to dir where dirname is the version id

            with ZipFile(zipped_filepath, 'r') as zipobj:
                zipobj.extractall(path=dir)
            
            tmp_file = os.path.join(dir, os.listdir(dir)[0])
            tmp_name = os.path.join(dir, f"{version}.txt")
            final_name = os.path.join(filepath, f"{version}.txt")

            os.rename(tmp_file, tmp_name) # Rename the file to <version>.txt
            shutil.move(tmp_name, final_name) # Move the file to parent dir
            shutil.rmtree(dir) # Clean up the temp dir
            os.unlink(zipped_filepath) # Delete zipped file
            
            print(f"SUCCESS: {version}.zip unzipped to \n{unzipped_filepath}")
        
        except Exception as e:

            print(f"ERROR: {version}.zip unzip unsuccessful. \nReason: {e}")

    # 4. Fetch MSig Enrichment data
    auth = "http://www.gsea-msigdb.org/gsea/j_spring_security_check"
    url = "http://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/7.5.1/h.all.v7.5.1.entrez.gmt"
    msig_filepath = os.path.join(working_dir, RAW_DIR, MSIG_FILENAME)

    msig_exists = os.path.exists(msig_filepath)

    headers = {
        "User-Agent":"node2vec/1.0",
        "Accept":"*/*",
        "Accept-Encoding":"deflate, br",
        "Connection":"keep-alive",
        "Content-type":"application/x-www-form-urlencoded"
    }

    data = {
        "j_username":email,
        "j_password":"password"
    }

    if not msig_exists:

        print("Fetching gene set data... ")
        try:

            print("Authenticating...")
            response = requests.post(auth, headers=headers, data=data)

            if not response.status_code == requests.codes.ok:

                response.raise_for_status()
            
            print(f"SUCCESS: Authentication successful - status {response.status_code}")
            
            print("Downloading gene set data...")
            headers["Cookie"] = response.request.headers["Cookie"]
            headers = [(k, v) for k, v in headers.items()] # Convert from dict to list of tuples for urllib

            opener = urllib.request.build_opener()
            opener.addheaders = headers
            urllib.request.install_opener(opener)
            _, header = urllib.request.urlretrieve(url, msig_filepath)

        except Exception as e:

            print(f"DOWNLOAD UNSUCCESSFUL gene_set.entrez.gmt \nReason {e}")

        print(header)
        print(f"SUCCESS: data downloaded successfully\n{msig_filepath}")
    
    else:

        print(f"File {msig_filepath} already exists. \nSkipping download...")
