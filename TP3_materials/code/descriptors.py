#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

# Display nice progress bars (non-standard package: pip install tqdm to install it)
from tqdm import tqdm


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

def local_PCA(points):
    """
    Computes PCA of points
    Inputs:
              points = (N x d) matrix where "N" is the number of points and "d" the dimension
    Returns:
         eigenvalues = (d) matrix
        eigenvectors = (d x d) matrix
    """
    N, d = points.shape

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    cov = (centered_points.T @ centered_points)/N

    return np.linalg.eigh(cov)


def neighborhood_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    print('Computing neighborhood_PCA')
    t0 = time.time()

    N, d = query_points.shape
    all_eigenvalues = np.zeros((N, d))
    all_eigenvectors = np.zeros((N, d, d))

    print('Initializing KD-Tree...', end='\r')
    tree = KDTree(cloud_points)

    print('Searching neighborhoods...', end='\r')
    ind = tree.query_radius(query_points, radius)

    print('Running PCA on all neighborhoods...', end='\r')
    for i, neighbors in tqdm(enumerate(ind), total=N, desc='Running PCA on all neighborhoods', leave=False):
        neighborhood = cloud_points[neighbors]
        all_eigenvalues[i], all_eigenvectors[i] = local_PCA(neighborhood)

    t1 = time.time()
    print('Done. Time elapsed: {:.0f}s.'.format(t1-t0))
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud
    eigenvalues, eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)

    eigenvalues = np.maximum(eigenvalues, 1e-10) # avoid numerical errors

    l1, l2, l3 = eigenvalues[:,2], eigenvalues[:,1], eigenvalues[:,0]

    sine = eigenvectors[:,2,0]

    verticality = 2*np.arcsin(sine)/np.pi
    linearity = 1 - l2/l1
    planarity = (l2 - l3)/l1
    sphericity = l3/l1

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = local_PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # YOUR CODE
        all_eigenvalues, all_eigenvectors = neighborhood_PCA(cloud, cloud, 0.5)

        normals = all_eigenvectors[:,:,0]

        write_ply('../data/Lille_with_normals.ply', [cloud, normals], ['x','y','z','nx','ny','nz'])

    # Features computation
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # YOUR CODE
        v, l, p, s = compute_features(cloud, cloud, 0.5)

        write_ply('../data/Lille_features.ply', [cloud, v, l, p, s], ['x','y','z','verticality','linearity','planarity','sphericity'])
