#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

def center_on_centroid(points):
    centroid = np.sum(points,axis=0) / len(points)
    points = points - centroid
    return points, centroid

def divide_scale(points, n):
    points = points / n
    return points

def rotation_matrix(t):
    theta = np.radians(t)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

def rotation_z(points, t):
    points = np.dot(points, rotation_matrix(t).T)
    return points

def recenter(points, center):
    return points + center

def translation_y(points, y):
    points[:,1] += y
    return points



# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #

    # Path of the file
    file_path = '../data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T
    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    # Get the scalar field which represent density as a vector
    density = data['scalar_density']

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #

    # Replace this line by your code
    transformed_points, old_center = center_on_centroid(points)
    transformed_points = divide_scale(transformed_points, 2)
    transformed_points = rotation_z(transformed_points, -90)
    transformed_points = recenter(transformed_points, old_center)
    transformed_points = translation_y(transformed_points, -0.1)

    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('../data/little_bunny.ply', [transformed_points, colors, density], ['x', 'y', 'z', 'red', 'green', 'blue', 'density'])

    print('Done')
