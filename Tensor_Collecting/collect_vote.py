import numpy as np
from tools import *
import copy

def initialize_tensor (img, projection_mat, n_neib, const, sigma, noise):

  kernel_size = 2 * n_neib + 1
  
  pad_img = np.pad(img, n_neib, mode="constant")
  pad_img_shape = np.shape(pad_img)

  initial_tensor_with_pad = np.zeros((pad_img_shape[0], pad_img_shape[1], 2, 2))

  for row in range(n_neib, pad_img_shape[0] - n_neib):
    for col in range(n_neib, pad_img_shape[1] - n_neib):

      # if some pixel values are low, we will lable it as background
      if pad_img[row][col] > noise:

        #neibs = copy.copy(pad_img[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1])
        #neibs_reshape = np.reshape(neibs, (kernel_size, kernel_size, 1, 1))
        
        #init_tensor = neibs_reshape * np.array([[1, 0], [0, 1]])
        #neib_vote = np.multiply(projection_mat, init_tensor)

        #initial_tensor_with_pad[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1] += neib_vote

        initial_tensor_with_pad[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1] += projection_mat
        
  
  return initial_tensor_with_pad



def collect_votes (mat, n_neib, cone_angle, const, sigma):
  
  kernel_size = 2 * n_neib + 1

  relation_mat = get_neib_relation_mat(n_neib)

  shape = np.shape(mat)
  accumulated_tensor_mat = copy.copy(mat)

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      #receivers = copy.copy(mat[row - n_neib : row + n_neib + 1, col - n_neib : col + n_neib + 1])      
      voter = copy.copy(mat[row][col])

      val, vec = np.linalg.eig(voter)

      if (val[0] - val[1]) != 0:
        
        norm_dir = get_degree(vec[0][0], vec[1][0])
    
        #print(norm_dir)
        # this is to 
        change_sign = False
        if norm_dir > 180:
          change_sign = True
          norm_dir -= 180
        
        tangent_dir = norm_dir + 90
        if norm_dir >= 90:
          tangent_dir = norm_dir - 90
        
        up_range = tangent_dir + cone_angle
        low_range = tangent_dir - cone_angle
        
        if up_range > 180:
            up_range -= 180

        if low_range < 0:
            low_range += 180

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
                
                projection = get_projection(degree, tangent_dir, r, const, sigma, change_sign)

                #cast_vote = projection * voter
                cast_vote = np.multiply(projection, voter)
                projection_mat[i][j] = cast_vote

                opp_y = n_neib + y
                opp_x = n_neib + x
                if i != opp_y:
                  if change_sign == False:
                    change_sign = True
                  else:
                    change_sign = False
                  opp_projection = get_projection(degree, tangent_dir, r, const, sigma, change_sign)
                  #opp_cast_vot = opp_projection * voter
                  opp_cast_vot = np.multiply(opp_projection, voter)
                  projection_mat[opp_y][opp_x] = opp_cast_vot

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

      eig_diff = abs(val[0]) - abs(val[1])
      #eig_diff = abs(val[0] - val[1])
      val_mat[row][col] = eig_diff

      e1 = np.array([vec[0][0], vec[1][0]])
      
      vec_mat[row][col] = eig_diff * np.outer(np.transpose(e1), e1)

 
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
  max_eigval = sort_list[-1] - threshold_point

  for row in range(n_neib, shape[0] - n_neib):
    for col in range(n_neib, shape[1] - n_neib):
      pix = val_mat[row][col]
      if pix < threshold_point:
        val_mat[row][col] = 0
      else:
        val_mat[row][col] = np.floor((pix - threshold_point) / max_eigval * 255)
  
  return val_mat, vec_mat
