#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Plane detection by region growing
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np
import os
import queue

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def neighborhood_PCA(query_points, cloud_points, radius):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    print('Computing neighborhood_PCA')

    N, d = query_points.shape

    print('Initializing KD-Tree...', end='\r')
    tree = KDTree(cloud_points)

    print('Searching neighborhoods...', end='\r')
    ind = tree.query_radius(query_points, radius)

    print('Running PCA on all neighborhoods...', end='\r')
    neighborhoods = [cloud_points[idx] for idx in ind]
    centered_points = [points - np.mean(points, axis=0) for points in neighborhoods]
    covariances = np.array([points.T @ points / len(points) for points in centered_points])
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)

    return eigenvalues, eigenvectors


def compute_planarities_and_normals(points, radius):

    eigenvalues, eigenvectors = neighborhood_PCA(points, points, radius)

    planarities = (eigenvalues[:,1] - eigenvalues[:,0])/eigenvalues[:,2]
    normals = eigenvectors[...,0]
    return planarities, normals


def region_criterion(p1, p2, n1, n2):
    r"""
        Returns whether (p1,n1) and (p2,n2) satisfy the requested criterion

        Inputs
        ------
        p1 : (N1 x d) array containing query points
        p2 : (1 x d) array containing reference points
        n1 : (N1 x d) array containing query normals
        n2 : (1 x d) array containing reference normals

        Returns
        -------
        (N1 x 1) boolean array indicating if the criterion is verified
    """
    norm1 = np.maximum(np.linalg.norm(n1, axis=-1), 1e-10)[:,np.newaxis] # shape: (N1 x 1)
    norm2 = np.maximum(np.linalg.norm(n2, axis=-1), 1e-10)[:,np.newaxis] # shape; (N2 x 1)

    # first criterion
    distance = np.abs((p1 - p2) @ n2.T) / norm2.T

    # second criterion
    # print((n1 @ n2.T) / (norm1 @ norm2.T))
    angle = np.arccos(np.clip((n1 @ n2.T) / (norm1 @ norm2.T), -1, 1))
    # print(angle)

    return (distance < THRESHOLD_1) * (np.abs(angle) < THRESHOLD_2)


def queue_criterion(p):
    return p > THRESHOLD_P


def RegionGrowing(cloud, normals, planarities, radius):
    N = len(cloud)
    region = np.zeros(N, dtype=bool)
    visited = np.zeros(N, dtype=bool)
    Q = queue.Queue()
    tree = KDTree(cloud)

    # seed = np.random.randint(N)
    seed = np.argmax(planarities)

    region[seed] = visited[seed] = True
    Q.put(seed)

    while not Q.empty():
        q = Q.get()
        point, normal = cloud[[q]], normals[[q]]
        # print(q, point, cloud[q], point.shape)
        neighbors = tree.query_radius(point, radius)[0]

        neighbors = neighbors[np.logical_not(visited[neighbors])]
        visited[neighbors] = True

        in_region = neighbors[region_criterion(cloud[neighbors], point, normals[neighbors], normal).squeeze()]
        region[in_region] = True

        in_queue = in_region[queue_criterion(planarities[in_region])]

        for n in in_queue:
            Q.put(n)

    return region


def multi_RegionGrowing(cloud, normals, planarities, radius, NB_PLANES=2):
    plane_labels = -np.ones(len(cloud), dtype=np.int32)

    for plane in range(NB_PLANES):
        print('Computing plane {:d}/{:d}...'.format(plane+1, NB_PLANES), end='\r')
        remaining_inds = np.flatnonzero(plane_labels==-1)

        region = RegionGrowing(
            cloud[remaining_inds],
            normals[remaining_inds],
            planarities[remaining_inds],
            radius
        )
        plane_labels[remaining_inds[region]] = plane

    plane_inds, remaining_inds = plane_labels>=0, plane_labels<0

    return plane_inds, remaining_inds, plane_labels[plane_inds]


# ----------------------------------------------------------------------------------------------------------------------
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
    planarities_path = '../data/planarities.ply'
    if os.path.exists(planarities_path):
        # Load point cloud
        data = read_ply(planarities_path)

        # Concatenate data
        points = np.vstack((data['x'], data['y'], data['z'])).T
        normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T

        planarities = data['planarities']

    else:
        # Path of the file
        file_path = '../data/indoor_scan.ply'

        # Load point cloud
        data = read_ply(file_path)

        # Concatenate data
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        N = len(points)

        # Computes normals of the whole cloud
        # ***********************************
        #

        # Parameters for normals computation
        radius = 0.2

        # Computes normals of the whole cloud
        t0 = time.time()
        planarities, normals = compute_planarities_and_normals(points, radius)
        t1 = time.time()
        print('normals and planarities computation done in {:.3f} seconds'.format(t1 - t0))


        # Save
        write_ply(planarities_path,
                  [points, colors, normals, planarities],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz', 'planarities'])

    # Find a plane by Region Growing
    # ******************************
    #

    if False:
        # Define parameters of Region Growing
        radius = 0.2
        # thresholds
        THRESHOLD_1 = 0.2
        THRESHOLD_2 = 10
        THRESHOLD_P = 0.9

        # Find a plane by Region Growing
        t0 = time.time()
        region = RegionGrowing(points, normals, planarities, radius)
        t1 = time.time()
        print('Region Growing done in {:.3f} seconds'.format(t1 - t0))

        # Get inds from bollean array
        plane_inds = region.nonzero()[0]
        remaining_inds = (1 - region).nonzero()[0]

        # Save the best plane
        # write_ply('../best_plane.ply',
        #           [points[plane_inds], colors[plane_inds], labels[plane_inds], planarities[plane_inds]],
        #           ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'planarities'])
        # write_ply('../remaining_points.ply',
        #           [points[remaining_inds], colors[remaining_inds], labels[remaining_inds], planarities[remaining_inds]],
        #           ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'planarities'])
        write_ply('../data/best_plane.ply',
                  [points[plane_inds], colors[plane_inds], np.zeros(len(plane_inds)), planarities[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'planarities'])
        write_ply('../data/remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds], planarities[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'planarities'])
        # write_ply('../data/best_plane.ply', [points, colors, region.astype(np.int32), planarities],
        #             ['x', 'y', 'z', 'red','green', 'blue', 'label', 'planarities'])
    # Find multiple in the cloud
    # ******************************
    #

    if True:
        os.makedirs('../results/RegionGrowing', exist_ok=True)
        # Define parameters of multi_RegionGrowing
        radius = 0.2
        NB_PLANES = 20
        # thresholds
        THRESHOLD_1 = 0.2
        THRESHOLD_2 = 10
        THRESHOLD_P = 0.95

        # Recursively find best plane by RegionGrowing
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RegionGrowing(points, normals, planarities, radius, NB_PLANES)
        t1 = time.time()
        print('multi RegionGrowing done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../results/RegionGrowing/best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../results/RegionGrowing/remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
