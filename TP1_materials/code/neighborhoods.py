#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    # YOUR CODE
    distances = cdist(queries, supports)
    close = distances < radius
    neighborhoods = np.array([supports[c] for c in close])
    return neighborhoods


def brute_force_KNN(queries, supports, k):
    # YOUR CODE
    distances = cdist(queries, supports)
    arg_neighborhoods = np.argpartition(distances, k, axis=-1)[:,:k]
    neighborhoods = supports[arg_neighborhoods]
    return neighborhoods





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
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('===== BRUTE FORCE =====')
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:
        # imports
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        # Define the search parameters
        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 1000

        # YOUR CODE
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        print('\n===== KD-TREE =====')

        r_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        leaf_size = 84

        t0 = np.zeros(len(r_list))
        t1 = np.zeros(len(r_list))

        tree = KDTree(points, leaf_size=leaf_size)

        for i, r in enumerate(r_list):
            t0[i] = time.time()
            radius_neighbors = tree.query_radius(queries, r)
            t1[i] = time.time()

            # Print timing results
            # print('radius:', r)
            # print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
            # print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))
            #
            # # Time to compute all neighborhoods in the cloud
            # total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
            # print('Computing spherical neighborhoods on whole cloud : {:.0f} minutes'.format(total_spherical_time / 60))
            # print('Computing KNN on whole cloud : {:.0f} minutes'.format(total_KNN_time / 60))

        plt.plot(r_list, t1-t0, c='b')
        plt.title('Computation time as a function of radius')
        plt.xlabel('radius (m)')
        plt.ylabel('time (s)')
        plt.show()

        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        print(total_spherical_time)
