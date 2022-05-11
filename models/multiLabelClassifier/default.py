import tensorflow as tf

class Model(tf.keras.Model):
    '''
    Useful resources:
    > https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
    > https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72
    '''

    def __init__(self, text_vectorizer, encoder, num_classes):
        '''
        '''

        super().__init__()

        self.text_vectorizer = text_vectorizer
        self.encoder = encoder
        self.embedding_dim = encoder.output_dim
        self.output_dim = num_classes

        self._create_fc_layers()

    def compile_model(self):
        '''
        '''
        self.compile(
            optimizer='adam',
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )        
    
    def call(self, input):
        '''
        '''
        x = self.text_vectorizer(input)
        x = self.encoder(x)
        x = self.fc_1(x)
        x = self.fc_2(x)

        # print(x)
        # print(tf.reshape(x, (50,)))
        # print(tf.transpose(tf.reshape(x, (50,))))
        # print(tf.reshape(x, (None,50)))

        # return tf.reshape(x, (50, 1))
        return tf.reshape(x, (50,))
        # return tf.transpose(tf.reshape(x, (50,)))
    
    def _create_fc_layers(self):
        '''
        '''

        self.fc_1 = tf.keras.layers.Dense(
            self.embedding_dim,
            activation='leaky_relu',
            use_bias=False            
        )

        self.fc_2 = tf.keras.layers.Dense(
            self.output_dim,
            activation='sigmoid',
        )        
       

