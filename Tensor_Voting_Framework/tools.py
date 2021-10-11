import numpy as np
import matplotlib.pyplot as plt


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



def a_get_projection (degree, r_dist, const, sigma):
  length = r_dist
  theta = degree

  # if theta is bigger than 360, it may create side effect
  if theta > 360:
    theta -= 360

  rad = theta * 3.1416 / 180
  s = np.sin(rad)

  # if theta in the range of 45 < θ < 135 or 225 < θ < 315
  status = False
  if (theta > 45 and theta < 135) or (theta > 225 and theta < 315):
    # switch status if θ is in the range
    status = True
    s = np.cos(rad)

  # update curve length if it is not a straight line
  if s != 0 and rad != 0:
    adj_degree = degree
    if adj_degree >= 180:
      adj_degree -= 180
    if adj_degree > 135:
      adj_degree -= 135
    if adj_degree > 90:
      adj_degree -= 90
    if adj_degree > 45:
      adj_degree -= 45
    
    adj_rad = adj_degree * 3.1416 / 180
    length = adj_rad * length / s
  
  k = 2 * s / length

  DF = np.exp(-(length**2 + const * k**2) / sigma**2)

  proj_rad = 2 * rad

  if status:
    tensor_vote = DF * np.array(
      [[(np.sin(proj_rad))**2, -np.sin(proj_rad) * np.cos(proj_rad)], 
      [-np.sin(proj_rad) * np.cos(proj_rad), (np.cos(proj_rad))**2]])
    return tensor_vote
  else:
    tensor_vote = DF * np.array(
      [[(np.cos(proj_rad))**2, -np.sin(proj_rad) * np.cos(proj_rad)], 
      [-np.sin(proj_rad) * np.cos(proj_rad), (np.sin(proj_rad))**2]])
    return tensor_vote
  



def get_projection (degree, tangent, r_dist, const, sigma, sign):
  length = r_dist

  rad = (degree - tangent) * 3.1416 / 180
  if sign == True:
    rad = -rad
  s = np.sin(rad)

  # if theta in the range of 45 < θ < 135 or 225 < θ < 315
  status = False
  if (degree > 45 and degree < 135) or (degree > 225 and degree < 315):
    # switch status if θ is in the range
    status = True
    s = np.cos(rad)

  # update curve length if it is not a straight line
  if s != 0 and rad != 0:
    length = rad * length / s
  
  k = 2 * s / length

  DF = np.exp(-(length**2 + const * k**2) / sigma**2)

  proj_rad = 2 * rad

  if status: 
    tensor_vote = DF * np.array(
      [[(np.sin(proj_rad))**2, -np.sin(proj_rad) * np.cos(proj_rad)], 
      [-np.sin(proj_rad) * np.cos(proj_rad), (np.cos(proj_rad))**2]])
    return tensor_vote
  else:
    tensor_vote = DF * np.array(
      [[(np.cos(proj_rad))**2, -np.sin(proj_rad) * np.cos(proj_rad)], 
      [-np.sin(proj_rad) * np.cos(proj_rad), (np.sin(proj_rad))**2]])
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
        
        tangent = round(degree / 45, 0) * 45

        projection_mat[pos] = get_projection(degree, tangent, r, const, sigma, False)
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



def visualize_direction(grid_h, grid_w, grid_span, vec_mat):
  x_pos = np.arange(0,grid_w,grid_span)
  y_pos = np.arange(0,grid_h,grid_span)

  X, Y = np.meshgrid(x_pos, y_pos)

  shape = np.shape(vec_mat)

  U = np.zeros((shape[0], shape[1]))
  V = np.zeros((shape[0], shape[1]))

  
  for row in range(0, shape[0]):
    for col in range(0, shape[1]):
      eig_val, eig_vec = np.linalg.eig(vec_mat[row][col])
      a = eig_vec[0][0]
      b = eig_vec[1][0]
      print(a, b)
      U[row][col] = a
      V[row][col] = b

  #fig, ax = plt.subplots()
  #ax.quiver(X, Y, U, V)

  plt.figure()
  plt.quiver(X, Y, U, V, units='width')

  plt.show()