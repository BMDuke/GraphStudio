import os
import importlib

import tensorflow as tf

from source.utils.config import Config
from source.utils.archive import Archive, ModelArchive

class Model(object):
    '''
    Thinsg to include
    > Tensorboard: Add callback, view logs
    > checkpoint
    > save model
    > restore model (from checkpoint/ from saved model)
    > Make and manage directories
    '''

    # Class attrs
    base_dir = 'cache/tensorflow/models'        # Model weights, checkpoints etc...
    data_dir = ''                               # Location of data
    walk_dir = ''                               # Location of walk dataset
    source_dir = ''                             # Source files for model architecture 
    default_class_name = 'Model'                # The class name given to architecture implementations

    def __init__(self):
        '''
        '''
        config = Config()
        self.config = config
        self.data_archive = Archive(config)
        self.model_archive = ModelArchive()

    def train(self):
        '''




        Step 1: Parse config 
        Step 2: Resolve dependencies
        Step 3: Load assets
        Step 4: Set up environment and models
        Step 5: Train
        Step 6: Save
        '''
    
    def test(self):
        '''
        '''

    def predict(self):
        '''
        '''

    def _get_callback_tensorboard(self, log_dir):
        '''
        '''
        callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_steps_per_second=True,
            update_freq='epoch'
        )

        return callback

    def _launch_tensorboard_(self, log_dir):
        '''
        '''
        os.system(f'tensorboard --logdir {log_dir}')


    def _get_callback_checkpoints(self, checkpoint_dir, monitor, save_best_only=False):
        '''
        '''        
        callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir,
            monitor=monitor,
            verbose=0,
            save_best_only=save_best_only,
            save_weights_only=True,
            save_freq='epoch'
        )      

        return callback

    def _restore_from_checkpoints(self, checkpoint_dir):
        '''
        '''    

        self.model.load_weights(checkpoint_dir)    # Test that this does not reauire reassignment
    
    def _save_model(self, model, filepath):
        '''
        '''

        try:

            tf.keras.models.save_model(
                model,
                filepath,
                overwrite=True,
                include_optimizer=True,
                save_format='tf'
            )
        
        except Exception as e:

            print(f'ERROR: {e}')
            raise
    
    def _load_model(self, filepath):
        '''
        '''

        try:
            
            model = tf.keras.models.load_model(filepath, compile=True)

            self.model = model

        except Exception as e:
            
            print(f'ERROR: {e}')
            raise            

    def _new_experiment(self, uuid):
        '''
        
        '''

        # Make experiment root dir
        experiment_root = os.path.join(self.base_dir, uuid)
        os.makedirs(experiment_root)

        # Create subdirs for logs, checkpoints and models
        log_dir = os.path.join(experiment_root, 'logs')
        checkpoints_dir = os.path.join(experiment_root, 'checkpoints')
        model_dir = os.path.join(experiment_root, 'model')

        os.mkdir(log_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(model_dir)

        return experiment_root, log_dir, checkpoints_dir, model_dir

    def _import_model(self, filepath):
        '''
        '''
        
        # Convert path/to/package => path.to.package
        module_path = os.path.splitext(filepath)[0].replace('/', '.')
        
        # Import module and extract model
        module = importlib.import_module(module_path)
        model = getattr(module, self.default_class_name)

        self.model = model

        return self.model        





