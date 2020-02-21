#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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

# display progress bars
from tqdm import tqdm

# Import functions from scikit-learn
from sklearn.preprocessing import label_binarize

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


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[0::factor]
    decimated_colors = colors[0::factor]
    decimated_labels = labels[0::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    # YOUR CODE
    print('Computing grid subsampling', end='\r')

    N, C = points.shape
    indices = points // voxel_size

    uniques, counts = np.unique(indices, axis=0, return_counts=True)

    # initialize non-empty voxels
    grid = {tuple(idx): 0 for idx in uniques}

    # fill non-empty voxels with the sum of points in it
    for idx, p in tqdm(zip(indices, points), desc='Computing grid subsampling', total=N, leave=False):
        grid[tuple(idx)] += p

    # divide by the number of points per voxel to get the barycenter
    for idx, count in zip(uniques, counts):
        grid[tuple(idx)] /= count

    # convert to numpy array
    return np.array(tuple(grid.values()))

def grid_subsampling_colors(points, colors, voxel_size, labels):

    # YOUR CODE
    print('Computing grid subsampling', end='\r')

    colors = np.float32(colors)

    indices = np.int32(points // voxel_size)
    N, C = indices.shape

    uniques, counts = np.unique(indices, axis=0, return_counts=True)

    distinct_labels = list(np.unique(labels))
    n_labels = len(distinct_labels)

    # initialize non-empty voxels
    subsampled_points = {tuple(idx): 0 for idx in uniques}
    subsampled_colors = {tuple(idx): 0 for idx in uniques}
    subsampled_labels = {tuple(idx): np.zeros(n_labels) for idx in uniques}

    # fill non-empty voxels with the sum of points in it
    for idx, p, c, l in tqdm(zip(indices, points, colors, labels), desc='Computing grid subsampling', total=N, leave=False):
        subsampled_points[tuple(idx)] += p
        subsampled_colors[tuple(idx)] += c
        subsampled_labels[tuple(idx)][distinct_labels.index(l)] += 1

    # divide by the number of points per voxel to get the barycenter
    for idx, count in zip(uniques, counts):
        subsampled_points[tuple(idx)] /= count
        subsampled_colors[tuple(idx)] /= count
        subsampled_labels[tuple(idx)] = distinct_labels[np.argmax(subsampled_labels[tuple(idx)])]

    # convert to numpy array
    return np.array(tuple(subsampled_points.values())), np.array(tuple(subsampled_colors.values()), dtype=np.uint8), np.array(tuple(subsampled_labels.values()))


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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../data/decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    GET_COLORS = True

    # Subsample
    if GET_COLORS:
        t0 = time.time()
        subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling_colors(points, colors, voxel_size, labels)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

        # Save
        write_ply('../data/grid_subsampled_colors.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    else:
        t0 = time.time()
        subsampled_points = grid_subsampling(points, voxel_size)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

        # Save
        write_ply('../data/grid_subsampled.ply', [subsampled_points], ['x', 'y', 'z'])

    print('Done')
