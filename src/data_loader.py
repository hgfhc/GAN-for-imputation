from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
from utils import binary_sampler


def data_loader (miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: UCI datasets or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
  data_x = mnist.train.images[0:50000]

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m