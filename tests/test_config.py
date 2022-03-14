import unittest
import yaml
import os
import pathlib

from source.utils.config import Config

filepath = "conf"
current = "__current__"
template = "__template__"

# Configure unittest
unittest.TestLoader.sortTestMethodsUsing = None

class TestConfig(unittest.TestCase):

    '''
    Testing suite for config 
    '''

    @classmethod
    def setUpClass(cls) -> None:
        
        cls.name = '__test__' # shared testing document
        cls.fp = filepath
        cls.current_fp = f"{current}.yaml"
        cls.template_fp = f"{template}.yaml"

        # Save current config
        cur = os.path.join(filepath, cls.current_fp)
        with open(cur, 'r') as f: 
            curr = yaml.load(f, Loader=yaml.FullLoader)          
        cls.pretest_config = curr['current'] # Save to restore after tests

        config = Config()

        config.new(cls.name) # Creates a conf.yaml file just for testing

        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:

        # Restore pre-test config
        cur = os.path.join(cls.fp, cls.current_fp)
        with open(cur, 'r') as f: 
            curr = yaml.load(f, Loader=yaml.FullLoader)     
        curr["current"] = cls.pretest_config     
        with open(cur, 'w') as f: 
            _ = yaml.dump(curr, f) # Restore pre-test config

        # Delete testing files
        name1 = '__test1__'
        name2 = '__test2__'
        config = Config()

        config.delete(cls.name, force=True)

        fp1 = pathlib.Path(os.path.join(cls.fp, f"{name1}.yaml"))
        fp2 = pathlib.Path(os.path.join(cls.fp, f"{name2}.yaml"))

        if fp1.exists():
            os.remove(fp1)

        if fp2.exists():
            os.remove(fp2)
        

        return super().tearDownClass()

    def setUp(self) -> None:

        # Make sure current config is testing config
        cur = os.path.join(self.fp, self.current_fp)
        with open(cur, 'r') as f: 
            curr = yaml.load(f, Loader=yaml.FullLoader)     
        curr["current"] = self.name     
        with open(cur, 'w') as f: 
            _ = yaml.dump(curr, f)

        return super().setUp()



    def test_1_new_config(self):
        '''
        Testing for:
        > Existence of file when created
        > Config values are the same as __template__
        > Name error is thrown if name already exists
        > The current config is swithced to the newly created config
        > There is no error if no parameters are passed
        > If parameters are passed, the config is updated. 
        '''

        config = Config()
        name1 = '__test1__'
        name2 = '__test2__'

        # Config file creation
        config.new(name1)
        config.new(name2, *['p=0.1', 'q=0.9']) # sys.argv data

        # Read in configs
        fp1 = os.path.join(self.fp, f"{name1}.yaml")
        fp2 = os.path.join(self.fp, f"{name2}.yaml")
        tmp = os.path.join(self.fp, self.template_fp)
        cur = os.path.join(self.fp, self.current_fp)

        with open(fp1, 'r') as f: # Unchanged
            conf1 = yaml.load(f, Loader=yaml.FullLoader)

        with open(fp2, 'r') as f: # Changed
            conf2 = yaml.load(f, Loader=yaml.FullLoader)

        with open(tmp, 'r') as f: # Template for comparison
            temp = yaml.load(f, Loader=yaml.FullLoader)

        with open(cur, 'r') as f: # Current config
            curr = yaml.load(f, Loader=yaml.FullLoader)   


        # ++++ Testing ++++
        self.assertTrue(pathlib.Path(fp1).exists()) # Succesfully created
        self.assertEqual(conf1, temp) # new config == template
        self.assertNotEqual(conf2, temp) # new config + params != template
        self.assertTrue(curr["current"] == name2) # current == newest config
        self.assertEqual(conf2['data']['p'], 0.1) # parameters set
        self.assertEqual(conf2['data']['q'], 0.9) # parameters set
        with self.assertRaises(NameError): # File already exists
            config.new(name1)


    def test_2_new_experiment(self):
        '''
        Testing for:
        > Accurate insertion of n2v configuration
        > Accurate insertion of mlc configuration
        > Accurate insertion of bc configuration

        ** Note all information is passed in as strings to 
            mimic command line input 
            In future this should be converted to appropriate
            datatype
        '''

        # Set up variables
        bc = 'binaryClassifier'
        mlc = 'multiLabelClassifier'
        n2v = 'node2vec'
        name = 'test_experiment'
        values = {
            'epochs': 1000,
            'encoder': 'tester',
            'split': 0.999,
            'embedding_dim': 2000,
        }
        # Mock commandline input
        inp_n2v = f"embedding_dim={values['embedding_dim']} epochs={values['epochs']}".split(' ')
        inp_mlc = f"split={values['split']} encoder={values['encoder']} epochs={values['epochs']}".split(' ')
        inp_bc = f"split={values['split']} encoder={values['encoder']} epochs={values['epochs']}".split(' ')

        config = Config()

        # Add experiments
        config.add_experiment(n2v, name, *inp_n2v)
        config.add_experiment(mlc, name, *inp_mlc)
        config.add_experiment(bc, name, *inp_bc)

        # Load config file
        fp1 = os.path.join(self.fp, f"{self.name}.yaml")
        with open(fp1, 'r') as f: 
            conf1 = yaml.load(f, Loader=yaml.FullLoader)        
        out_n2v =  conf1[n2v]['experiments']
        out_mlc = conf1[mlc]['experiments']
        out_bc = conf1[bc]['experiments']

        # ++++ Testing ++++
        self.assertEqual(out_n2v[name]['epochs'], values['epochs'])
        self.assertEqual(out_n2v[name]['embedding_dim'], values['embedding_dim'])
        self.assertEqual(out_mlc[name]['epochs'], values['epochs'])
        self.assertEqual(out_mlc[name]['encoder'], values['encoder'])
        self.assertEqual(out_mlc[name]['split'], values['split'])
        self.assertEqual(out_bc[name]['epochs'], values['epochs'])
        self.assertEqual(out_bc[name]['encoder'], values['encoder'])
        self.assertEqual(out_bc[name]['split'], values['split'])

    def test_3_use_config(self):
        '''
        Testing for:
        > Accurate assignment of config name
        > Error handling when name not found
        '''

        filename = '__current__'

        config = Config()

        config.set(filename)

        cur = os.path.join(self.fp, self.current_fp)
        with open(cur, 'r') as f: 
            curr = yaml.load(f, Loader=yaml.FullLoader)         

        # ++++ Testing ++++
        self.assertEqual(curr['current'], filename) # Accurate assignment
        with self.assertRaises(NameError): # File doesnt exists
            config.set('non-existent-config')        

    def test_4_use_experiment(self):
        '''
        Testing for:
        > Accurate assignment of experiment name
        > Error handling when name not found
        '''        

        bc = 'binaryClassifier'
        mlc = 'multiLabelClassifier'
        n2v = 'node2vec'
        name = 'test_experiment'
        error = 'missing_name'

        config = Config()

        config.set_experiment(bc, name)
        config.set_experiment(mlc, name)
        config.set_experiment(n2v, name)

        fp1 = os.path.join(self.fp, f"{self.name}.yaml")
        with open(fp1, 'r') as f: 
            curr = yaml.load(f, Loader=yaml.FullLoader) 

        # ++++ Testing ++++
        self.assertEqual(curr[bc]['current'], name) # Accurate assignment
        self.assertEqual(curr[mlc]['current'], name) # Accurate assignment
        self.assertEqual(curr[n2v]['current'], name) # Accurate assignment
        with self.assertRaises(AssertionError): # File doesnt exists
            config.set_experiment(n2v, error)

    def test_5_show(self):
        '''
        Testing for:
        > Returns current config when no args are passed
        > Returns named config when name is passed
        > Returns error when name is not found
        '''        
        name = template
        error = 'missing_name'

        config = Config()

        curr_conf = config.show()
        named_conf = config.show(name)

        cur = os.path.join(self.fp, self.current_fp)
        with open(cur, 'r') as f: 
            curr = yaml.load(f, Loader=yaml.FullLoader)        

        fp1 = os.path.join(self.fp, f"{curr['current']}.yaml") # Current conf
        with open(fp1, 'r') as f: 
            conf1 = yaml.load(f, Loader=yaml.FullLoader) 

        fp2 = os.path.join(self.fp, f"{name}.yaml") # Template conf
        with open(fp2, 'r') as f: 
            conf2 = yaml.load(f, Loader=yaml.FullLoader)                          

        # ++++ Testing ++++
        self.assertEqual(curr_conf, conf1) # Accurate retreival
        self.assertEqual(named_conf, conf2) # Accurate retreival
        with self.assertRaises(NameError): # File doesnt exists
            named_conf = config.show(error) 

    def test_6_current(self):
        '''
        Testing for:
        > Confirm the correct config is returned
        '''        

        # Setup function sets current to self.name
        config = Config()

        curr = config.current()

        # ++++ Testing ++++
        self.assertEqual(curr, self.name) # Accurate retreival

    def test_7_edit_config(self):
        '''
        Testing for:
        > Accurate update of config values
        > Raise error if config name not found
        '''        

        name = self.name
        error = 'missing_name'
        values = {
            'p': 1000,
            'q': -10,
            'version': '1.1.111'
        }        

        config = Config()

        fp1 = os.path.join(self.fp, f"{name}.yaml") # Conf before edit
        with open(fp1, 'r') as f: 
            conf1 = yaml.load(f, Loader=yaml.FullLoader)         
        
        config.edit(name, *[f"{k}={v}" for k, v in values.items()])

        with open(fp1, 'r') as f: 
            conf2 = yaml.load(f, Loader=yaml.FullLoader) # conf after edit

        # ++++ Testing ++++
        self.assertNotEqual(conf1, conf2) # Config has been modified
        self.assertEqual(conf2['data']['p'], values['p']) # Accurate assignment
        self.assertEqual(conf2['data']['q'], values['q']) # Accurate assignment
        self.assertEqual(conf2['data']['version'], values['version']) # Accurate assignment
        with self.assertRaises(NameError): # File doesnt exists
            config.edit(error, *[f"{k}={v}" for k, v in values.items()])


    def test_8_edit_experiment(self):
        '''
        Testing for:
        > Experiment conf has changed
        > Experiment conf values are accurate
        > Raise error if experiment name not found
        '''        

        # Set up variables
        bc = 'binaryClassifier'
        mlc = 'multiLabelClassifier'
        n2v = 'node2vec'
        name = 'test_experiment'
        error = 'missing_name'
        values = {
            'epochs': 1,
            'encoder': 'siesta',
            'split': 0.0001
        }
        values_n2v = {
            'epochs': 1,
            'embedding_dim': 3.14
        }        

        config = Config()

        fp1 = os.path.join(self.fp, f"{self.name}.yaml")
        with open(fp1, 'r') as f: 
            conf1 = yaml.load(f, Loader=yaml.FullLoader)  # Conf before edit

        config.edit_experiment(bc, name, *[f"{k}={v}" for k, v in values.items()])  
        config.edit_experiment(mlc, name, *[f"{k}={v}" for k, v in values.items()])  
        config.edit_experiment(n2v, name, *[f"{k}={v}" for k, v in values_n2v.items()])  

        with open(fp1, 'r') as f: 
            conf2 = yaml.load(f, Loader=yaml.FullLoader)  # Conf after edit              

        out_n2v =  conf2[n2v]['experiments']
        out_mlc = conf2[mlc]['experiments']
        out_bc = conf2[bc]['experiments']

        # ++++ Testing ++++
        self.assertNotEqual(conf1, conf2) # Config has been modified
        self.assertEqual(out_n2v[name]['epochs'], values_n2v['epochs']) # Accurate assignment
        self.assertEqual(out_n2v[name]['embedding_dim'], values_n2v['embedding_dim']) # Accurate assignment
        self.assertEqual(out_mlc[name]['epochs'], values['epochs']) # Accurate assignment
        self.assertEqual(out_mlc[name]['encoder'], values['encoder']) # Accurate assignment
        self.assertEqual(out_mlc[name]['split'], values['split']) # Accurate assignment        
        self.assertEqual(out_bc[name]['epochs'], values['epochs']) # Accurate assignment
        self.assertEqual(out_bc[name]['encoder'], values['encoder']) # Accurate assignment
        self.assertEqual(out_bc[name]['split'], values['split']) # Accurate assignment        
        with self.assertRaises(AssertionError): # File doesnt exists
            config.edit_experiment(n2v, error, *[f"{k}={v}" for k, v in values.items()])  

    def test_9_delete_config(self):
        '''
        Testing for:
        > Absence of file after delete
        > Error if file does not exist
        '''

        config = Config()
        name1 = '__test1__'
        name2 = '__test2__'

        config.delete(name1, force=True)
        config.delete(name2, force=True)

        # Make filepaths
        fp1 = os.path.join(self.fp, f"{name1}.yaml")
        fp2 = os.path.join(self.fp, f"{name2}.yaml")       

        # ++++ Testing ++++
        self.assertFalse(pathlib.Path(fp1).exists()) # Succesfully deleted
        self.assertFalse(pathlib.Path(fp2).exists()) # Succesfully deleted   
        with self.assertRaises(NameError): # File does not exist
            config.delete(name1)              


    def test_99_delete_experiment(self):
        '''
        Testing for:
        > Conf has changed
        > Experiment not found after delete
        > Error if name does not exist
        '''        
        
        # Set up variables
        bc = 'binaryClassifier'
        mlc = 'multiLabelClassifier'
        n2v = 'node2vec'
        name = 'test_experiment'

        config = Config()

        fp1 = os.path.join(self.fp, f"{self.name}.yaml")
        with open(fp1, 'r') as f: 
            conf1 = yaml.load(f, Loader=yaml.FullLoader)  # Conf before edit

        config.delete_experiment(bc, name)
        config.delete_experiment(mlc, name)
        config.delete_experiment(n2v, name)

        with open(fp1, 'r') as f: 
            conf2 = yaml.load(f, Loader=yaml.FullLoader)  # Conf after edit    

        # ++++ Testing ++++
        self.assertNotEqual(conf1, conf2) # Config has been modified
        with self.assertRaises(AssertionError): # Successfully deleted & file does not exist
            config.delete_experiment(bc, name)             
        with self.assertRaises(AssertionError): # Successfully deleted & file does not exist
            config.delete_experiment(mlc, name)     
        with self.assertRaises(AssertionError): # Successfully deleted & file does not exist
            config.delete_experiment(n2v, name)                                 





if __name__ == "__main__":
    TestConfig.main()