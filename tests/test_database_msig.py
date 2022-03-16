import unittest
import os

import numpy as np
import pandas as pd

from source.datasets.msig import MSig

# Configure unittest
unittest.TestLoader.sortTestMethodsUsing = None

class TestMSig(unittest.TestCase):

    '''
    Testing suite for MSig dataset 
    * Set verbose to false so not to disrupt the printing process
    > ✓ self._load_raw_msig()
    > ✓ self._parse_gmt(raw)
    > ✓ self._make_test_filepath()
    > ✓ self._save()
    > ✓ self._load_msig()
    > ✓ self.validate()
    

    '''    

    @classmethod
    def setUpClass(cls) -> None:
        
        # Load and save first and last lines of msigdb
        # Count the number of genes in those lines

        # Set path variables
        cls.raw_dir = 'data/raw'
        cls.processed_dir = 'data/processed/gene_ids'
        cls.msig_filename = "gene_set.entrez.gmt"
        cls.out_filename = "gene_ids_labelled.csv"
        cls.header1 = 'function' 
        cls.header2 = 'gene_id' 

        # Check raw MSig dataset and load 
        msig = os.path.join(cls.raw_dir, cls.msig_filename)

        if os.path.exists(msig):

            try: 
                with open(msig, 'r') as f: 
                    lines = []
                    while True:
                        line = f.readline()
                        if not line:    # EOF
                            break
                        lines.append(line)
            except Exception as e:
                print(f"ERROR: {e}")

        else:             
            print(f"ERROR: {msig} not found\nHave you downloaded the MSig dataset?") 

        # Calculate some descriptions of first and last lines of the dataset
        first_line = lines[0]
        first_line_tokens = first_line.split('\t')
        first_line_class = first_line_tokens[0]
        first_line_num_genes = len(first_line_tokens[2:])

        last_line = lines[-1]
        last_line_tokens = last_line.split('\t')
        last_line_class = last_line_tokens[0]
        last_line_num_genes = len(last_line_tokens[2:])        

        # Add information to class
        cls.num_rows = len(lines)

        cls.first = {
            'raw':first_line,
            'tokens': first_line_tokens,
            'class': first_line_class,
            'count': first_line_num_genes
        }

        cls.last = {
            'raw':last_line,
            'tokens': last_line_tokens,
            'class': last_line_class,
            'count': last_line_num_genes
        }        
        
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:

        msig = MSig(debug=True, verbose=False)

        fp = os.path.join(cls.processed_dir, cls.out_filename)
        fp = msig._make_test_filepath(fp)

        os.remove(fp)
        
        return super().tearDownClass()

    def setUp(self) -> None:

        return super().setUp()



    def test_1_load_raw_data(self):
        '''
        Testing for:
        > Load does not throw error
        > Accuracy: raw content match for 
            > first line
            > last line
        > Accuracy:
            > Num lines match
        '''

        msig = MSig(debug=True, verbose=False)

        raw = msig._load_raw_msig()  
        first_line = raw[0]
        last_line = raw[-1]

        # ++++ Testing ++++
        self.assertEqual(first_line, self.first['raw']) # Raw content matches
        self.assertEqual(last_line, self.last['raw']) # Raw content matches
        self.assertEqual(len(raw), self.num_rows) # Data is the same size

    def test_2_parse_gmt(self):
        '''
        Testing for:
        > URL is removed:
            > processed data has len(raw) -1
            > No field starts with 'http'
        > There are no NA's
        > Has shape (2, count-1)
        '''

        msig = MSig(debug=True, verbose=False)

        raw = msig._load_raw_msig()  
        first_line = msig._parse_gmt([self.first['raw'], self.last['raw']])

        # ++++ Testing ++++
        for value in first_line[self.header2]:
            self.assertNotRegex(str(value), 'http')
            self.assertIsInstance(value, int)
            self.assertIsNot(value, None)

        self.assertEqual(
            len(first_line[self.header2]), 
            self.first['count'] + self.last['count'])

    def test_3_make_test_filepath(self):
        '''
        Testing for:
        > Accuracy:
            > Filepath is modified in expected ways 
            > Filepath is modified for complicated inputs
            > Filepath is handles error
        '''

        msig = MSig(debug=True, verbose=False)

        file1 = '/this/is/a/test.csv'
        file2 = '/this/is/a/harder.test.csv'
        file3 = 'test.csv'
        file4 = 'test'
        file5 = ''

        expected_output1 = '/this/is/a/test_TEST.csv'
        expected_output2 = '/this/is/a/harder_TEST.test.csv'
        expected_output3 = 'test_TEST.csv'
        expected_output4 = 'test_TEST'

        output1 = msig._make_test_filepath(file1)
        output2 = msig._make_test_filepath(file2)
        output3 = msig._make_test_filepath(file3)
        output4 = msig._make_test_filepath(file4)

        # ++++ Testing ++++
        self.assertEqual(output1, expected_output1) # Modification is accuate
        self.assertEqual(output2, expected_output2) # Modification is accuate
        self.assertEqual(output3, expected_output3) # Modification is accuate
        self.assertEqual(output4, expected_output4) # Modification is accuate
        with self.assertRaises(AssertionError): # File doesnt exists
            msig._make_test_filepath(file5)

    def test_4_save(self):
        '''
        Testing for:
        > Save works
        '''

        msig = MSig(debug=True, verbose=False)

        fp = os.path.join(self.processed_dir, self.out_filename)
        fp = msig._make_test_filepath(fp)
        raw = self.first['raw']
        processed = msig._parse_gmt([raw])

        df_out = pd.DataFrame(data=processed)

        msig._save(df_out)

        df_in = pd.read_csv(fp)

        # ++++ Testing ++++
        self.assertTrue(df_out.equals(df_in)) # Save is successful

    def test_5_load(self):
        '''
        Testing for:
        > load works
        '''

        msig = MSig(debug=True, verbose=False)

        raw = self.first['raw']
        processed = msig._parse_gmt([raw])

        df = pd.DataFrame(data=processed)

        df_in = msig._load_msig()

        # ++++ Testing ++++
        self.assertTrue(df.equals(df_in)) # Save is successful   

    def test_5_validate(self):
        '''
        Testing for:
        > Validates correct data
        > Throws error for invalid data
        '''

        msig = MSig(debug=True, verbose=False)

        raw = self.first['raw']

        def invalidate():
            processed = msig._parse_gmt([raw])
            df = pd.DataFrame(data=processed)
            df.iloc[[3]] = np.NAN
            msig._save(df)

        # df_in = msig._load_msig()

        # ++++ Testing ++++
        self.assertTrue(msig.validate()) # Initial data is valid
        invalidate()
        self.assertFalse(msig.validate()) # Invalid data detected





if __name__ == "__main__":
    TestMSig.main()
