import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, InputLayer

class DataPrep:
  """class to prepare the training data by reshaping it, changing the type and normalizing it"""
  def __init__(self,data):
      self.data = data

  def dataprep(self):
      self.data = self.data.reshape(self.data.shape[0],28,28,1)
      self.data = self.data.astype('float32')
      self.data /= 255
      print('data shape:', self.data.shape)
      print('Number of images in data:', self.data.shape[0])
      print('max of data:', np.max(self.data), 'min of data:', np.min(self.data))
      print('data type:', type(self.data))     
        
        
class LabelsPrep:
  """class to prepare the labels as one-hot vectors"""
	def __init__(self, data, num_classes):
	    self.labels = data
	    self.num_classes = num_classes
	   
	def labelsprep(self):
	    self.labels = tf.keras.utils.to_categorical(self.labels, num_classes = self.num_classes)
	    print('labels data shape:', self.labels.shape)
	    print('labels data type:', type(self.labels))


def CNN_dropout_hidden_fun(input_shape):
  """define the CNN model"""
  model = Sequential()
  model.add(InputLayer(input_shape = input_shape))
  model.add(Dropout(0.3))
  model.add(Conv2D(256, kernel_size = (3,3),activation = tf.nn.relu))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Conv2D(128, kernel_size = (3,3),activation = tf.nn.relu))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
  model.add(Dense(100, activation = tf.nn.relu))
  model.add(Dense(100,activation = tf.nn.relu))
  model.add(Dense(10,activation = tf.nn.softmax))
  return model        


class CNN:
  """class to fit a CNN"""
  PARAMETERS = {'loss':"categorical_crossentropy", 
                'optimizer': "adam", 
                'metrics': "accuracy", 
                'epochs': 10, 
                'batch_size': 1000, 
                'shuffle': True
               }
    
  def __init__(self, 
               data_train, 
               labels_train, 
               data_test, 
               labels_test, 
               input_shape,
               parameters = PARAMETERS, 
              ):
      self.data_train = data_train 
      self.labels_train = labels_train
      self.data_test = data_test
      self.labels_test = labels_test
      self.params = parameters
      self.input_shape = input_shape
     
  def trainmodel(self):
    """train and compile the model"""
    self.CNN_dropout_hidden = CNN_dropout_hidden_fun(self.input_shape)
    self.CNN_dropout_hidden.compile(loss = self.params['loss'], 
                                    optimizer = self.params['optimizer'], 
                                    metrics = self.params['metrics']
                                   )

  def accuracy(self):
    """print accuracy history per epoch"""
    self.history_dropout_hidden = self.CNN_dropout_hidden.fit(self.data_train, 
                                                              self.labels_train, 
                                                              validation_data = (self.data_test, self.labels_test), 
                                                              epochs = self.params['epochs'],
                                                              batch_size = self.params['batch_size'],
                                                              shuffle = self.params['shuffle']
                                                             )
    self.scores_dropout_hidden = self.CNN_dropout_hidden.evaluate(self.data_test,
                                                                  self.labels_test
                                                                 )    
    print("Accuracy: %.2f%%" %(self.scores_dropout_hidden[1]*100))
 
  def plottrain(self):
    """plot loss and accuracy against epoch for training data"""
    plt.subplot(121)
    plt.plot(self.history_dropout_hidden.history['accuracy'])

    plt.subplot(122)
    plt.plot(self.history_dropout_hidden.history['loss'])
    
  def plotval(self):
    """plot loss and accuracy against epoch for validation data"""
    plt.subplot(121)
    val_accuracy = plt.plot(self.history_dropout_hidden.history['val_accuracy'])

    plt.subplot(122)
    var_loss = plt.plot(self.history_dropout_hidden.history['val_loss'])
    return val_accuracy, val_loss
      
        