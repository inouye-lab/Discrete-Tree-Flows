import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
from scipy.stats import multivariate_normal
import torch
torch.manual_seed(25)
import itertools

def preprocess_805_snp_data(snps_805_path):
    df = pd.read_csv(snps_805_path, sep = ' ', header=None, compression='infer')
    df = df.sample(frac=1, random_state = 42).reset_index(drop=True)
    df_noname = df.drop(df.columns[0:2], axis=1)
    return df_noname.values
   
def process_mushroom_data(mushroom_data_path):
  mushroom_data = pd.read_csv(mushroom_data_path,header=None)
  for col_id in mushroom_data.columns:
    unique_col_data = np.unique(mushroom_data.iloc[:,col_id])
    mapper = {}
    for i,d in enumerate(unique_col_data):
      mapper[d] = i
    mushroom_data.iloc[:,col_id] = [ mapper[x] for x in mushroom_data.iloc[:,col_id]]
    unique_col_data = np.unique(mushroom_data.iloc[:,col_id])
  mushroom_data = mushroom_data.drop([11],axis = 1) #11 had missing data, so I dropped it
  
  #return mushroom_data.to_numpy()
  return mushroom_data.values
def preprocess_cop(cop_data_path):
  return np.load(cop_data_path).astype(int)

def preprocess_digits_binarize(threshold):
  digits = load_digits()
  digits.data[digits.data < threshold] = 0
  digits.data[digits.data >= threshold] = 1
  return digits.data.astype(int)  

def preprocess_binary_mnist():
  train,test = tfds.load('binarized_mnist', split=['train', 'test'])
  new_train = tfds.as_numpy(train)
  new_test = tfds.as_numpy(test)
  flattened_images = []
  for i,ex in enumerate(new_train):
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    flattened_images.append(ex['image'].flatten())
  for i,ex in enumerate(new_test):
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    flattened_images.append(ex['image'].flatten())
  return np.array(flattened_images).astype(int)
 
#from: https://github.com/TrentBrick/PyTorchDiscreteFlows/blob/master/Figure2Replication.ipynb 
def sample_quantized_gaussian_mixture(batch_size):
    """Samples data from a 2D quantized mixture of Gaussians.
    This is a quantized version of the mixture of Gaussians experiment from the
    Unrolled GANS paper (Metz et al., 2017).
    Args:
        batch_size: The total number of observations.
    Returns:
        Tensor with shape `[batch_size, 2]`, where each entry is in
            `{0, 1, ..., max_quantized_value - 1}`, a rounded sample from a mixture
            of Gaussians.
    """
    clusters = np.array([[2., 0.], [np.sqrt(2), np.sqrt(2)],
                                             [0., 2.], [-np.sqrt(2), np.sqrt(2)],
                                             [-2., 0.], [-np.sqrt(2), -np.sqrt(2)],
                                             [0., -2.], [np.sqrt(2), -np.sqrt(2)]])
    assignments = torch.distributions.OneHotCategorical(
            logits=torch.zeros(8, dtype = torch.float32)).sample([batch_size])
    means = torch.matmul(assignments, torch.from_numpy(clusters).float())

    samples = torch.distributions.normal.Normal(loc=means, scale=0.1).sample()
    clipped_samples = torch.clamp(samples, -2.25, 2.25)
    quantized_samples = (torch.round(clipped_samples * 20) + 45).long()
    return quantized_samples
