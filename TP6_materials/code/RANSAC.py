#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
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


def compute_plane(points):

    p1, p2, p3 = points
    normal = np.cross(p2 - p1, p3 - p1)

    return p1, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):

    norm = np.maximum(np.linalg.norm(normal), 1e-10)
    distance = np.abs((points - ref_pt) @ normal) / norm

    return np.flatnonzero(distance<threshold_in)


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):

    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    best_nb = 0

    for i in range(NB_RANDOM_DRAWS):
        # Select random set of 3 points
        random_indices = np.random.choice(len(points), 3, replace=False)
        random_points = points[random_indices]

        # Evaluate random plane
        ref_pt, normal = compute_plane(random_points)
        nb_points_in_plane = len(in_plane(points, ref_pt, normal, threshold_in))

        if nb_points_in_plane > best_nb :
            # Update best plane
            best_ref_pt = ref_pt
            best_normal = normal
            best_nb = nb_points_in_plane

    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    plane_labels = -np.ones(len(points), dtype=np.int32)

    for plane in range(NB_PLANES):
        print('Computing plane {:d}/{:d}...'.format(plane+1, NB_PLANES), end='\r')
        remaining_inds = np.flatnonzero(plane_labels==-1)

        ref_pt, normal = RANSAC(points[remaining_inds], NB_RANDOM_DRAWS, threshold_in)

        plane_labels[in_plane(points, ref_pt, normal, threshold_in)] = plane

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

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    N = len(points)

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if False:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]

        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if False:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:
        os.makedirs('../results/RANSAC', exist_ok=True)
        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 200
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../results/RANSAC/best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../results/RANSAC/remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
