#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from utils.visu import show_ICP, plot_RMS


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#
def mean_squared_error(y_true, y_pred, squared=True):
    '''
    Computes the mean-squared error
    Inputs :
        y_true = (d x N) matrix where "N" is the number of points and "d" the dimension
        y_pred = (d x N) matrix where "N" is the number of points and "d" the dimension
    '''

    mse = y_true.shape[0]*np.mean((y_true - y_pred)**2)
    return mse if squared else np.sqrt(mse)




def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    data_mean = np.mean(data, axis=1)[:,np.newaxis]
    ref_mean = np.mean(ref, axis=1)[:,np.newaxis]

    centered_data = data - data_mean
    centered_ref = ref - ref_mean

    U, S, Vt = np.linalg.svd(centered_data @ centered_ref.T)

    # compute rotation matrix
    R = (U @ Vt).T
    # force positive determinant
    if np.linalg.det(R) < 0:
        U[:,-1] = -U[:,-1]
        R = (U @ Vt).T

    # compute translation vector
    T = ref_mean - R @ data_mean

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration

    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    d = data.shape[0]
    old_R, old_T = np.eye(d), np.zeros((d,1))
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE
    tree = KDTree(ref.T)

    for _ in range(max_iter):
        # find nearest neighbors
        # print(tree.query(data_aligned.T))
        ind = np.squeeze(tree.query(data_aligned.T)[1])

        # compute best transform
        ref_points = ref.T[ind].T
        R, T = best_rigid_transform(data_aligned, ref_points)

        # align data
        data_aligned = R @ data_aligned + T

        # add results to lists
        R, old_R = R @ old_R, R
        T, old_T = T + old_T, T
        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(ind)

        # stopping criterion
        RMS = mean_squared_error(ref_points, data_aligned, squared=False)
        RMS_list.append(RMS)
        # print(RMS)
        if RMS < RMS_threshold:
            break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_stochastic(data, ref, max_iter, RMS_threshold, sampling_limit):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration

    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    d, N = data.shape
    RMS_list = []

    # YOUR CODE
    tree = KDTree(ref.T)

    for _ in range(max_iter):
        # find nearest neighbors
        random_idx = np.random.choice(N, sampling_limit, replace=False)
        sampled_data = data_aligned[:,random_idx]
        ind = np.squeeze(tree.query(sampled_data.T)[1])

        # compute best transform
        ref_points = ref.T[ind].T
        R, T = best_rigid_transform(sampled_data, ref_points)

        # align data
        data_aligned = R @ data_aligned + T

        # stopping criterion
        RMS = mean_squared_error(ref_points, data_aligned[:,random_idx], squared=False)
        RMS_list.append(RMS)
        # print(RMS)
        if RMS < RMS_threshold:
            break

    return RMS_list


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':

    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

        # Load clouds
        data = read_ply(bunny_o_path)
        bunny_original = np.vstack((data['x'], data['y'], data['z']))

        data = read_ply(bunny_r_path)
        bunny_returned = np.vstack((data['x'], data['y'], data['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_returned, bunny_original)

        print('R:\n', R)
        print('T:\n', T)

        # Apply the tranformation
        bunny_aligned = R @ bunny_returned + T

        # Save cloud
        write_ply('../data/bunny_transformed.ply', bunny_aligned.T, ['x','y','z'])

        # Compute RMS
        rms = mean_squared_error(bunny_original, bunny_aligned, squared=False)

        # Print RMS
        print('RMS: {:.3e}'.format(rms))


    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds
        # Load clouds
        ref = read_ply(ref2D_path)
        ref = np.vstack((ref['x'], ref['y']))

        data = read_ply(data2D_path)
        data = np.vstack((data['x'], data['y']))

        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data, ref, 20, 1e-6)

        # Show ICP
        show_ICP(data_aligned, ref, R_list, T_list, neighbors_list)

        # plot RMS
        plot_RMS(RMS_list)

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'

        # Load clouds
        ref = read_ply(bunny_o_path)
        ref = np.vstack((ref['x'], ref['y'], ref['z']))

        data = read_ply(bunny_p_path)
        data = np.vstack((data['x'], data['y'], data['z']))

        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data, ref, 50, 1e-6)

        # Show ICP
        show_ICP(data_aligned, ref, R_list, T_list, neighbors_list)

        # plot RMS
        plot_RMS(RMS_list, c='b')


    # Fast ICP
    # ********
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        ref = read_ply(NDDC_1_path)
        ref = np.vstack((ref['x'], ref['y'], ref['z']))

        data = read_ply(NDDC_2_path)
        data = np.vstack((data['x'], data['y'], data['z']))

        sampling_limits = [1000, 10000, 50000]
        colors = 'rgb'

        for i, sampling_limit in enumerate(sampling_limits):
            print('[{}/{}] Computing stochastic ICP with a sampling limit of {} points'.format(i+1, len(sampling_limits), sampling_limit))
            # Apply ICP
            RMS_list = icp_point_to_point_stochastic(data, ref, 100, 1e-6, sampling_limit)

            # plot RMS
            plot_RMS(RMS_list, label=sampling_limit, c=colors[i] if colors else None, show=False)

        print('Done.')
        plt.show()
