import tensorflow as tf
import numpy as np
from tqdm import tqdm

from GIN import GIN
from LIN import LIN
from utils import xavier_init

def Guider(data_m, miss_data_x, gagin_parameters):
      
    no, dim = data_m.shape
    idx = [
      (0, no//2, 0, dim//2),
      (0, no//2, dim//2, dim),
      (no//2, no, 0, dim//2),
      (no//2, no, dim//2, dim)
    ]
    cnt_array = [data_m[idx[x][0]:idx[x][1], idx[x][2]:idx[x][3]].sum() for x in range(4)]
    min_idx = cnt_array.index(min(cnt_array))

    # Impute missing data
    imputed_data_x = GIN(miss_data_x, gagin_parameters)

    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [None, dim])
    
    #net variables
    
    N_W1 = tf.Variable(xavier_init([dim, dim]))  
    N_b1 = tf.Variable(tf.zeros(shape = [dim]))
    
    N_W2 = tf.Variable(xavier_init([dim, dim]))
    N_b2 = tf.Variable(tf.zeros(shape = [dim]))

    N_W3 = tf.Variable(xavier_init([dim, dim]))
    N_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
    theta = [N_W1, N_W2, N_W3, N_b1, N_b2, N_b3]
  
    ##functions
    # net
    def net(x):
      N_h1 = tf.nn.relu(tf.matmul(x, N_W1) + N_b1)
      N_h2 = tf.nn.relu(tf.matmul(N_h1, N_W2) + N_b2)   
      N_prob = tf.nn.sigmoid(tf.matmul(N_h2, N_W3) + N_b3)
      return N_prob
  
    ##structure
    N_sample = net(X)
               
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    X_mb = imputed_data_x            
    imputed_data_x = sess.run([N_sample], feed_dict = {X: X_mb})[0]
    
            
    imputed_submat = imputed_data_x[idx[min_idx][0]:idx[min_idx][1], idx[min_idx][2]:idx[min_idx][3]]
    datam_submat = data_m[idx[min_idx][0]:idx[min_idx][1], idx[min_idx][2]:idx[min_idx][3]]
    imputed_submat[datam_submat == 0] = np.nan
    print(imputed_submat)
    
    
    imputed_data_x_local = LIN(imputed_submat, gagin_parameters)
    imputed_data_x[idx[min_idx][0]:idx[min_idx][1], idx[min_idx][2]:idx[min_idx][3]] = imputed_data_x_local
    
    return imputed_data_x