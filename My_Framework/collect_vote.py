import numpy as np
from tools import *
import copy


def initialize_tensor (img, projection_mat, n_neib, const, sigma, noise):

  kernel_size = 2 * n_neib + 1
  
  pad_img = np.pad(img, n_neib, mode="constant")
  shape = np.shape(pad_img)

  accumulated_tensor_mat = np.zeros((shape[0], shape[1], 2, 2))

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      # if i am 0, should i collect?
      if pad_img[row][col] > noise:
        receivers = pad_img[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1]
        receivers_reshape = np.reshape(receivers, (kernel_size, kernel_size, 1, 1))

        all_projection = receivers_reshape * projection_mat

        all_projection_reshape = np.reshape(all_projection, (-1, 2, 2))
        summa = sum(all_projection_reshape)

        accumulated_tensor_mat[row][col] = summa

  #val_mat, vec_mat = eigen_decompose(accumulated_tensor_mat, n_neib)
  return accumulated_tensor_mat
  


def collect_votes (mat, projection_mat, n_neib, const, sigma):
  
  kernel_size = 2 * n_neib + 1

  shape = np.shape(mat)
  accumulated_tensor_mat = np.zeros(shape)

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):

      receivers = copy.copy(mat[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1])

      all_projection = np.multiply(receivers, projection_mat)

      all_projection_reshape = np.reshape(all_projection, (-1, 2, 2))

      accumulated_tensor_mat[row][col] = sum(all_projection_reshape)

  return accumulated_tensor_mat           



def eigen_decompose (mat, n_neib):
  shape = np.shape(mat)
  val_mat = np.zeros((shape[0], shape[1]))
  vec_mat = np.zeros((shape[0], shape[1], 2, 2))

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      t = mat[row][col]
      val, vec = np.linalg.eig(t)
      
      eig_diff = abs(abs(val[0]) - abs(val[1]))
      val_mat[row][col] = eig_diff

      vec_mat[row][col] = vec
  return val_mat, mat



def threshold (vals, vecs, n_neib, percent):
  shape = np.shape(vals)
  val_mat = vals
  vec_mat = vecs

  unique_list = np.unique(val_mat)
  sort_list = np.sort(unique_list)

  list_length = np.size(sort_list)
  print(list_length)

  threshold_point = sort_list[ int(np.ceil(list_length * percent)) ]
  max_eigval = sort_list[-1] - threshold_point

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      pix = val_mat[row][col]
      if pix < threshold_point:
        val_mat[row][col] = 0
      else:
        val_mat[row][col] = np.floor((pix - threshold_point) / max_eigval * 255)
  
  return val_mat, vec_mat
