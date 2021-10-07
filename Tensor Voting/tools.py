import numpy as np


def get_degree (x, y):
  degree = 361
  if x != 0:
    if y != 0:
      rad = np.arctan(y / x)
      degree = np.round(rad * 180 / 3.1416)
      
      if degree < 0:
        if y > 0: degree += 180
        else: degree += 360
      else:
        if y < 0: degree += 180
    
    else:
      if x > 0: degree = 0
      else: degree = 180
  else:
    if y < 0: degree = 270
    elif y > 0: degree = 90
  
  return degree



def get_neib_distance (const, sigma):
  decay_value = 1
  n_neib = 0

  while decay_value > 0.1:
    n_neib += 1
    decay_value = np.exp(-(const * n_neib) / sigma**2)
  
  return n_neib



def get_projection (degree, r_dist, const, sigma):
  length = r_dist

  rad = degree * 3.1416 / 180
  s = np.sin(rad)

  if s != 0 and rad != 0:
    length = rad * length / s
  
  k = 2 * s / length

  DF = np.exp(-(length**2 + const * k**2) / sigma**2)

  proj_rad = 2 * rad
  
  tensor_vote = DF * np.array(
    [[(np.sin(proj_rad))**2, -np.sin(proj_rad) * np.cos(proj_rad)], 
     [-np.sin(proj_rad) * np.cos(proj_rad), (np.cos(proj_rad))**2]])

  return tensor_vote



def get_projection_mat (n_neib, const, sigma):

  kernel_size = 2 * n_neib + 1

  projection_mat = np.zeros((kernel_size**2, 2, 2))
  pos = 0

  for y in range(-n_neib, n_neib + 1):
    for x in range(-n_neib, n_neib + 1):
      r = np.sqrt(x**2 + y**2)
      if r != 0:
        degree = get_degree(x, -y)
        if degree > 180:
          degree -= 180
        if degree > 90:
          degree -= 90
        if degree > 45:
          degree -= 45
          
        projection_mat[pos] = get_projection(degree, r, const, sigma)
      pos += 1

  projection_mat = np.reshape(projection_mat, (kernel_size, kernel_size, 2, 2))

  return projection_mat



def get_neib_relation_mat (n_neib):
  kernel_size = 2 * n_neib + 1
  relation_mat = np.zeros((kernel_size**2))
  pos = 0

  for y in range(-n_neib, n_neib + 1):
    for x in range(-n_neib, n_neib + 1):
      relation_mat[pos] = get_degree(x, -y)
      pos += 1

  relation_mat = np.reshape(relation_mat, (kernel_size, kernel_size))

  return relation_mat

