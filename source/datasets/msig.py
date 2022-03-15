import os

import pandas as pd
from prettytable import PrettyTable

class MSig(object):

    '''
    This class handles file IO, data exploration, data preprocessing
    and basic validation for the MGSig gene annotation dataset:
     - http://www.gsea-msigdb.org/gsea/index.jsp

    About:
    It handles the conversion of raw gene annotation data which is 
    in gmt format to standard tabular format with rows of 
    [gene_id, functional_annotation] pairs. This is output as a .csv
    file which can then be used to create the labelled edges dataset
    for supervised learning tasks. 

    Public API:
     - process()       Process the raw gmt data     
     - describe()      Print summary of the processed data
     - validate()      Validate the processed data
     - head()          Print the top n rows of the processed data

    Relationship to Vertices dataset:
    The output of this class is the input into the Vertices class 
    which creates a labelled Tensorflow dataset. 
    They have been seperated to deliniate a separation of 
    responsibilities:
     - MSig: process raw data, transform into tabular format
        of binary [gene_id, functional_annotation] pairs
     - Vertices: create tensorflow dataset, optimise dataset
        for training, represent as a table of one hot encoded
        examples of [gene_id, 0, 1, 0, 0, 1, ....] for supervised 
        learning tasks. 
    
    Useful resources:
     - https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats#GMT:_Gene_Matrix_Transposed_file_format_.28.2A.gmt.29
     - http://www.gsea-msigdb.org/gsea/index.jsp

    '''

    # Class attrs
    RAW_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed/gene_ids'

    MSIG_FILENAME = "gene_set.entrez.gmt"
    OUT_FILENAME = "gene_ids_labelled.csv"  

    HEADER1 = 'function'   
    HEADER2 = 'gene_id'   

    def __init__(self, config=None, debug=False, verbose=True):
        self.config = config # Include this for polymorphic interface
        self.debug = debug
        self.verbose = verbose # Can be disabled for unit testing

    def process(self):
        '''
        Orchestrates the processing of the raw MSig data
        '''

        raw = self._load_raw_msig()
        data = self._parse_gmt([raw])

        df = pd.DataFrame(data=data)

        is_valid = self.validate(df)
        assert is_valid, "ERROR: MSig.process() - dataset failed validation"

        self._save(df)

        return df

    def describe(self):
        '''
        Print out a description of the processed data 
        '''

        data = self._load_msig()

        # Class labels - HEADER1
        unique_labels = data[self.HEADER1].unique()
        unique_label_count = len(unique_labels)
        label_dist = data.groupby(self.HEADER1)

        # Data - HEADER2
        unique_proteins = data[self.HEADER2].unique()
        unique_protein_count = len(unique_proteins)
        protein_dist = data.groupby(self.HEADER2).count()    

        # Summary table
        num_rows = PrettyTable()
        num_rows.field_names = ['# rows', '# labels', '# genes']
        num_rows.add_row( [ len(data),  unique_label_count, unique_protein_count] )

        # Number of genes per label
        labels = PrettyTable()
        labels.field_names = ['Class labels', 'Number of examples w class']
        for k, v in label_dist:
            labels.add_row( [ k, len(v) ] )

        # Number of labels per gene
        multilable = PrettyTable()
        multilable.field_names = ['# class labels per gene', '# genes']
        duplicated_counts = data[self.HEADER2].value_counts()
        duplicate_distribution = duplicated_counts.value_counts()
        is_multilable_classification = False
        for index, value in duplicate_distribution.items():
            if index > 1:
                is_multilable_classification = True            
            multilable.add_row( [ index, value ] )

        # Display Summaries
        if self.verbose:
            print("+++++++++++++++++ SUMMARY +++++++++++++++++")
            print(num_rows, '\n')
            print("+++++++++++++++++ GENES PER LABEL +++++++++++++++++")
            print(labels, '\n') 
            print("+++++++++++++++++ LABELS PER GENE +++++++++++++++++")
            print(multilable, '\n')
            if is_multilable_classification:
                print('>> Learning is multilabel classification problem\n')
            else:
                print('>> Learning is multiclass classification problem\n')

    def validate(self, data=None):
        '''
        Validate the processed data:
         - Print the result
         - Return True if passes checks

        Checks:
         - No NA's
        '''
        if not data:
            data = self._load_msig()

        validation = PrettyTable()
        validation.field_names = ['Criteria', 'Result']

        # Check for NA's
        na_ceck = data.isnull().any()
        has_nas = False
        for header, value in na_ceck.items():
            validation.add_row( [f'Column "{header}" contains null values', value ] )
            if value:
                has_nas = True
        
        # Display validation results
        if self.verbose:
            print(validation)

        return not any([ has_nas ]) # Return True is all checks pass

    def head(self, nrows=10):
        '''
        Print the top n rows of the dataset
        '''    
        data = self._load_msig()

        if self.verbose:
            print(data.head(n=nrows))


    def _load_msig(self):
        '''
        Load the processed df
        '''
        filepath = os.path.join(self.PROCESSED_DIR, self.OUT_FILENAME)

        if self.debug:
            filepath = self._make_test_filepath(filepath)
        
        try:

            dataframe = pd.read_csv(filepath)
        
        except Exception as e: 

            print(f"ERROR: {e}")

        return dataframe

    def _load_raw_msig(self):
        '''
        Load the .gmt file that was downloaded from msigdb
        '''
        filepath = os.path.join(self.RAW_DIR, self.MSIG_FILENAME)
        
        try: 

            with open(filepath, 'r') as file_in:
                
                lines = []

                while True:
                    line = file_in.readline()
                    if not line:    # EOF
                        break
                    lines.append(line)

                return lines                     

        except Exception as e:

            print(f"ERROR: {e}")


    def _save(self, dataframe):
        '''
        Save the processed df
        '''
        filepath = os.path.join(self.PROCESSED_DIR, self.OUT_FILENAME)

        if self.debug:
            filepath = self._make_test_filepath(filepath)
        
        try:

            dataframe.to_csv(filepath, index=False)
            if self.verbose:
                print(f'SAVED: {filepath}')
        
        except Exception as e: 

            print(f"ERROR: {e}")
    

    def _make_test_filepath(self, filepath):
        '''
        Modify a filepath so that a copy can be used exclusively 
        for testing purposes without risking corrupting project data
        '''

        assert filepath 

        split_fp = filepath.split('.')
        split_fp[0] += '_TEST'

        return '.'.join(split_fp)

    def _parse_gmt(self, raw_lines):     
        '''
        Gmt format is:
        [annotation, url, gene1, gene2, gene3...]

        This transforms gmt to:
        [annotation, gene1]
        [annotation, gene2]

        Returns the data as a dict suitable for pandas 
        '''

        # Column headers as first entries in list to which values 
        # will be appended
        column1 = [self.HEADER1]
        column2 = [self.HEADER2]

        for line in raw_lines:
                
            tokens = line.split('\t')
                        
            # Line reads function, url, gene1, gene2, ...
            # We want the function and genes, not the url
            functional_annotation = tokens[0]    

            for idx in range(2, len(tokens)):
                gene = tokens[idx]

                # remove any newline characters
                if not gene.isdigit():
                    gene = gene.replace('\n', '')

                column1.append(functional_annotation)
                column2.append(int(gene))     

        # Transform into a dictionary and turn into a pandas DF
        data = {
            column1[0]: column1[1:],
            column2[0]: column2[1:]
        }       

        return data


        


if __name__ == "__main__":
    msig = MSig('config', debug=True)
    # gmt = msig.process()  # PASSED
    # print(gmt)    # PASSED
    # df = msig._load_msig()    # PASSED
    # print(df) # PASSED
    # msig.describe()   # PASSED
    # t = msig.validate()   # PASSED
    # print(t)    # PASSED
    # msig.head()
