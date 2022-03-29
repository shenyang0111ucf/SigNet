# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:24:27 2017

@author: sounak_dey and anjan_dutta
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint

#from keras.utils.visualize_util import plot

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from SignatureDataGenerator import SignatureDataGenerator
import getpass as gp
import sys
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import random
random.seed(1337)
import tensorflow_addons as tfa

# Create a session for running Ops on the Graph.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


class ModelCheckpointH5(ModelCheckpoint):
    # There is a bug saving models in TF 2.4
    # https://github.com/tensorflow/tensorflow/issues/47479
    # This forces the h5 format for saving
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 options=None,
                 **kwargs):
        super(ModelCheckpointH5, self).__init__(filepath,
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=False,
                                                save_weights_only=False,
                                                mode='auto',
                                                save_freq='epoch',
                                                options=None,
                                                **kwargs)

    def _save_model(self, epoch, logs):
        from tensorflow.python.keras.utils import tf_utils

        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options,
                                                save_format="h5")  # NK edited here
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options,
                                        save_format="h5")  # NK edited here

                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise


class LRN(tf.keras.layers.Layer):
    def __init__(self):
        super(LRN, self).__init__()
        self.depth_radius = 5
        self.bias = 2
        self.alpha = 1e-4
        self.beta = 0.75
    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(tf.cast(y_true, tf.float32) * K.square(y_pred) + (1 - tf.cast(y_true, tf.float32)) * K.square(K.maximum(margin - y_pred, 0)))
    
def create_base_network_signet(input_shape):
    print(input_shape)
    seq = Sequential()
    seq.add(Conv2D(96, (11, 11), activation='relu', name='conv1_1', strides=(1, 1), input_shape= input_shape,
                        kernel_initializer = 'glorot_uniform', bias_initializer='zeros', data_format='channels_last'))
    #seq.add(layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    seq.add(LRN())
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
    seq.add(ZeroPadding2D((2, 2), data_format='channels_last'))
    
    seq.add(Conv2D(256, (5, 5), activation='relu', name='conv2_1', strides=(1, 1), kernel_initializer = 'glorot_uniform', bias_initializer='zeros',  data_format='channels_last'))
    #seq.add(layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    seq.add(LRN())
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1), data_format='channels_last'))
    
    seq.add(Conv2D(384, (3, 3), activation='relu', name='conv3_1', strides=(1, 1), kernel_initializer = 'glorot_uniform', bias_initializer='zeros',  data_format='channels_last'))
    seq.add(ZeroPadding2D((1, 1), data_format='channels_last'))
    
    seq.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2', strides=(1, 1), kernel_initializer = 'glorot_uniform', bias_initializer='zeros', data_format='channels_last'))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
#    model.add(SpatialPyramidPooling([1, 2, 4]))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer='zeros'))
    seq.add(Dropout(0.5))
    
    seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer='zeros')) # softmax changed to relu

    return seq

#layers for vit
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = {
            "patch_size" : self.patch_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = {
            "num_patches" : self.num_patches,
            "projection" : self.projection,
            "position_embedding" : self.position_embedding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


data_augmentation = Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.experimental.preprocessing.Resizing(155, 220),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

def create_vit_classifier(inputs):
    img_height = 155
    img_width = 220
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (img_height // patch_size) * (img_width // patch_size)
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
    # Augment data.

    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    return features

def compute_accuracy_roc(predictions, labels):
   '''Compute ROC accuracy with a range of thresholds on distances.
   '''
   dmax = np.max(predictions)
   dmin = np.min(predictions)
   nsame = np.sum(labels == 1)
   ndiff = np.sum(labels == 0)
   
   step = 0.01
   max_acc = 0
   
   for d in np.arange(dmin, dmax+step, step):
       idx1 = predictions.ravel() <= d
       idx2 = predictions.ravel() > d
       
       tpr = float(np.sum(labels[idx1] == 1)) / nsame       
       tnr = float(np.sum(labels[idx2] == 0)) / ndiff
       acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
       if (acc > max_acc):
           max_acc = acc

   return max_acc
    
def read_signature_data(dataset, ntuples, height = 30, width = 100):
    
    usr = gp.getuser()

#    image_dir = '/home/' + usr + '/Workspace/SignatureVerification/Datasets/' + dataset + '/'
    image_dir = '/home/' + usr + '/github/SigNet/Datasets/' + dataset + '/'
    data_file = image_dir + dataset + '_pairs.txt'
    
    f = open( data_file, 'r' )
    lines = f.readlines()
    f.close()

    
    
    idx = np.random.choice(list(range(len(lines))), ntuples)
    
    lines = [lines[i] for i in idx]
    
    images = []
    
    for line in lines:
        file1, file2, label = line.split(' ')
                                       
        img1 = image.load_img(image_dir + file1, grayscale = True, 
                target_size=(height, width))
                
        img1 = image.img_to_array(img1)#, dim_ordering='tf')
                
        images.append(img1)
        
        img2 = image.load_img(image_dir + file1, grayscale = True, 
                target_size=(height, width))
            
        img2 = image.img_to_array(img2)#, dim_ordering='tf')
                
        images.append(img2)
        
    return np.array(images)
        
def main(args):
    dataset = args.dataset
    if dataset == 'Bengali':
    
        tot_writers = 100
        num_train_writers = 50
        num_valid_writers = 10
        
    elif dataset == 'Hindi':
        
        tot_writers = 160
        num_train_writers = 100
        num_valid_writers = 10
        
    elif dataset == 'GPDS300':
    
        tot_writers = 300
        num_train_writers = 240
        num_valid_writers = 30
        
    elif dataset == 'GPDS960':
    
        tot_writers = 4000
        num_train_writers = 3200
        num_valid_writers = 400
        
    elif dataset == 'CEDAR1':
    
        tot_writers = 55
        num_train_writers = 45
        num_valid_writers = 5
    
    num_test_writers = tot_writers - (num_train_writers + num_valid_writers)
    
    # parameters
    batch_sz = args.batch_size #128
    nsamples = args.num_samples #276 
    img_height = 155
    img_width = 220

    featurewise_center = False
    featurewise_std_normalization = True
    zca_whitening = False
    nb_epoch = args.epoch #20    
    input_shape=(img_height, img_width, 1)
    #parameters just for vit
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (img_height // patch_size) * (img_width // patch_size)
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    # initialize data generator   
    datagen = SignatureDataGenerator(dataset, tot_writers, num_train_writers, 
        num_valid_writers, num_test_writers, nsamples, batch_sz, img_height, img_width,
        featurewise_center, featurewise_std_normalization, zca_whitening)
    
    # data fit for std
    X_sample = read_signature_data(dataset, int(0.5*tot_writers), height=img_height, width=img_width)
    datagen.fit(X_sample)
    del X_sample
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_a = Input(shape=(input_shape))
        input_b = Input(shape=(input_shape))
        # network definition
        if args.model == 'cnn':
            base_network = create_base_network_signet(input_shape)
            processed_a = base_network(input_a)
            processed_b = base_network(input_b)
        elif args.model == 'vit':
            processed_a = create_vit_classifier(input_a)
            processed_b = create_vit_classifier(input_b)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        # processed_a = base_network(input_a)
        # processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        model = Model([input_a, input_b], distance)
        # compile model
        if args.model == 'cnn':
            opt = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=5e-4)
        elif args.model == 'vit':
            opt = tfa.optimizers.AdamW(
            learning_rate=0.002, weight_decay=0.0001
        )
        #adadelta = Adadelta()
        model.compile(loss=contrastive_loss, optimizer=opt)
    
    # display model
#    plot(model, show_shapes=True)
#    sys.exit()     
    
    # callbacks
    fname = os.path.join('/home/sh153803/github/SigNet/experiments/' , 'weights6_'+str(dataset)+'.h5')#
#     fname = '/home/sounak/Desktop/weights_GPDS300.hdf5'
    checkpointer = ModelCheckpointH5(filepath=fname, verbose=1, save_best_only=True)
#    tbpointer = TensorBoard(log_dir='/home/adutta/Desktop/Graph', histogram_freq=0,
#          write_graph=True, write_images=True)
    #print int(datagen.samples_per_valid)
    # print datagen.samples_per_train
    # print int(datagen.samples_per_valid)
    # print int(datagen.samples_per_test)
    # sys.exit()
    # train model   
    # model.fit_generator(generator=datagen.next_train(), samples_per_epoch=datagen.samples_per_train, nb_epoch=nb_epoch,
    #                     validation_data=datagen.next_valid(), nb_val_samples=int(datagen.samples_per_valid))   # KERAS 1

    model.fit_generator(generator=datagen.next_train(), steps_per_epoch=600, epochs=nb_epoch,
                        validation_data=datagen.next_valid(), validation_steps=120, callbacks=[checkpointer])  # KERAS 2
    #model.save_weights(fname)
    # 10% = steps*batch_size=69*80=46*120=23*240
    # load the best weights for test
    #model.load_weights(fname)
    print (fname)
    print ('Loading the best weights for testing done...')
   

#    tr_pred = model.predict_generator(generator=datagen.next_train(), val_samples=int(datagen.samples_per_train))
    # steps = numberofimages/batch_size
    te_pred = model.predict_generator(generator=datagen.next_test(), steps=480)
#    tr_acc = compute_accuracy_roc(tr_pred, datagen.train_labels)
    te_acc = compute_accuracy_roc(te_pred, datagen.test_labels)
    
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
# Main Function    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Signature Verification')
    # required training parameters
    parser.add_argument('--dataset', '-ds', action='store', type=str, required=True,
                  help='Please mention the database.')
    # required tensorflow parameters
    parser.add_argument('--epoch', '-e', action='store', type=int, default=20, 
                  help='The maximum number of iterations. Default: 20')
    parser.add_argument('--num_samples', '-ns', action='store', type=int, default=276, 
                  help='The number of samples. Default: 276')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=138,
                  help='The mini batch size. Default: 138')
    parser.add_argument('--model', '-ml', action='store', type=str, default='cnn')
    args = parser.parse_args()
    # print args.dataset, args.epoch, args.num_samples, args.batch_size
#    sys.exit()
    main(args)
