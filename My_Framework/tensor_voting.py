import numpy as np
from collect_vote import *
from tools import *


def iterative_tv (img, sigma, percent, iteration, noise):
  const = -16 * np.log(0.1) * (sigma - 1) / 3.1416**2

  n_neib = get_neib_distance(const, sigma)

  projection_mat = get_projection_mat(n_neib, const, sigma)
 
  vec_mat = initialize_tensor(img, projection_mat, n_neib, const, sigma, noise)
  
  val_mat = img

  while iteration > 0:
    tensor_mat = collect_votes(vec_mat, projection_mat, n_neib, const, sigma)
    val_mat, vec_mat = eigen_decompose(tensor_mat, n_neib)
    val_mat, vec_mat = threshold(val_mat, vec_mat, n_neib, percent)
    iteration -= 1
  
  size = np.shape(val_mat)
  result_val = val_mat[n_neib : size[0] - n_neib, n_neib : size[1] - n_neib]
  result_vec = vec_mat[n_neib : size[0] - n_neib, n_neib : size[1] - n_neib]
  return result_val, result_vec




def tensor_voting (img, cone_angle, sigma, percent, iteration, noise):

  const = -16 * np.log(0.1) * (sigma - 1) / 3.1416**2

  n_neib = get_neib_distance(const, sigma)

  projection_mat = get_projection_mat(n_neib, const, sigma)
  
  tensor_mat = initialize_tensor(img, projection_mat, n_neib, const, sigma, noise)
  
  collect_votes(tensor_mat, n_neib, cone_angle, const, sigma, percent)
  return "ok"