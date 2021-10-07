import numpy as np
from tools import *
import copy

def initialize_tensor (img, projection_mat, n_neib, const, sigma, noise):

  kernel_size = 2 * n_neib + 1
  
  pad_img = np.pad(img, n_neib, mode="constant")
  shape = np.shape(pad_img)

  initial_tensor_with_pad = np.zeros((shape[0], shape[1], 2, 2))

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):

      if pad_img[row][col] > noise:
        initial_tensor_with_pad[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1] += projection_mat
  return initial_tensor_with_pad



def initialize_tensor_with_pix (img, projection_mat, n_neib, const, sigma):

  kernel_size = 2 * n_neib + 1
  
  pad_img = np.pad(img, n_neib, mode="constant")
  shape = np.shape(pad_img)

  initial_tensor_with_pad = np.zeros((shape[0], shape[1], 2, 2))

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      tensor_proj = np.multiply(projection_mat, pad_img[row][col])
      initial_tensor_with_pad[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1] += tensor_proj

  return initial_tensor_with_pad



def collect_votes (mat, n_neib, cone_angle, const, sigma):
  
  kernel_size = 2 * n_neib + 1

  relation_mat = get_neib_relation_mat(n_neib)

  degrees_of_neib = np.zeros(kernel_size**2)

  shape = np.shape(mat)
  accumulated_tensor_mat = copy.copy(mat)

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      receivers = mat[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1]      
      voter = receivers[n_neib][n_neib]

      val, vec = np.linalg.eig(voter)

      if (val[0] - val[1]) != 0:
        norm_dir = get_degree(vec[0][0], vec[0][1])
        
        if norm_dir > 180:
          norm_dir -= 180
        
        tangent_dir = norm_dir + 90
        if norm_dir >= 90:
          tangent_dir = norm_dir - 90
        
        up_range = tangent_dir + cone_angle
        low_range = tangent_dir - cone_angle
        
        if up_range > 180:
            up_range = up_range - 180

        if low_range < 0:
            low_range = low_range + 180

        projection_mat = np.zeros((kernel_size, kernel_size, 2, 2))
        
        for i in range(0, int((kernel_size + 1) / 2)):
          for j in range(0, kernel_size):
            degree = relation_mat[i][j]
            if (up_range > low_range and degree >= low_range and degree <= up_range) or (low_range > up_range and (degree >= low_range or degree <= up_range)):

              y = n_neib - i
              x = n_neib - j
              r = np.sqrt(x**2 + y**2)

              if r != 0:
                if low_range > up_range:
                  if degree >= low_range:
                    if (tangent_dir - cone_angle) < 0:
                      degree = degree - 180
                  elif degree <= up_range:
                    if (tangent_dir + cone_angle) > 180:
                      degree = degree + 180
                
                
                projection = get_projection(degree - tangent_dir, r, const, sigma)

                #cast_vote = np.multiply(projection, voter)
                cast_vote = projection * voter
                projection_mat[i][j] = cast_vote

                opp_y = n_neib + y
                opp_x = n_neib + x
                if i != opp_y:
                  projection_mat[opp_y][opp_x] = cast_vote
        #print(projection_mat)
        #print("==============================")
        accumulated_tensor_mat[(row - n_neib) : (row + n_neib + 1), (col - n_neib) : (col + n_neib + 1)] += projection_mat
  return accumulated_tensor_mat           



def eigen_decompose (mat, n_neib):
  shape = np.shape(mat)
  val_mat = np.zeros((shape[0], shape[1]))
  vec_mat = np.zeros((shape[0], shape[1], 2, 2))

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      t = mat[row][col]
      val, vec = np.linalg.eig(t)

      eig_diff = abs(val[0] - val[1])
      val_mat[row][col] = eig_diff
      e1 = np.array([vec[0][0], vec[0][1]])

      vec_mat[row][col] = np.outer(np.transpose(e1), e1)
      #e2 = np.array([vec[1][0], vec[1][1]])
      #vec_mat[row][col] = eig_diff * np.outer(np.transpose(e1), e1) + val[1] * (np.outer(np.transpose(e1), e1) + np.outer(np.transpose(e2), e2))
  
  return val_mat, vec_mat



def threshold (vals, vecs, n_neib, percent):
  shape = np.shape(vals)
  val_mat = vals
  vec_mat = vecs

  unique_list = np.unique(val_mat)
  sort_list = np.sort(unique_list)

  list_length = np.size(sort_list)
  print(list_length)

  threshold_point = sort_list[ int(np.ceil(list_length * percent)) ]
  max_eigval = sort_list[-1]

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      pix = val_mat[row][col]
      if pix < threshold_point:
        val_mat[row][col] = 0
        vec_mat[row][col] = np.zeros((2,2))
      else:
        val_mat[row][col] = np.floor(pix / max_eigval * 255)
        #val_mat[row][col] = 255
  
  return val_mat, vec_mat
