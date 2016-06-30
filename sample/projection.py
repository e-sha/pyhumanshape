import numpy as np


"""
Converts homogeneous coordinates to heterogeneous ones
* in_pointArray is the input array of points. Each column is a single point (ndarray)

Retruns heterogeneous coordinates of the input points. Each column is a single
point.
"""
def Hom2Het(in_pointArray):
    return in_pointArray[:-1, :] / in_pointArray[-1, :]

"""
Converts heterogeneous coordinates to homogeneous ones
* in_pointArray is the input array of points. Each column is a single point (ndarray)

Retruns homogeneous coordinates of the input points. Each column is a single
point.
"""
def Het2Hom(in_pointArray):
    nPoints = in_pointArray.shape[1]
    return np.concatenate((in_pointArray, np.ones((1, nPoints))))

"""
Projects points from image coordinates to world coordinates on the specified plane
* in_matrix     is the input projection matrix (ndarray)
* in_pointArray is the input array of points. Each column is a single point (ndarray)
* in_plane      is the input projection plane in the row format (ndarray)

Retruns homogeneous coordinates of the input points on the specified plane. Each column is a single
point.
"""
def Im2World(in_matrix, in_pointArray, in_plane):
    if (in_plane.size != 4):
        raise ValueError("The input plane should have 4 values")
    if (in_pointArray.shape[0] == 2):
        # heterogeneous coordinates
        in_pointArray = Het2Hom(in_pointArray)
    if (in_pointArray.shape[0] != 3):
        raise ValueError("The input points should be 2D points stored column-wise")
    if (np.any(in_matrix.shape != np.array([3, 4]))):
        raise ValueError("Wrong size of the projection matrix. It should be 3-by-4 matrix")
    # X is the result iff in_plane.dot(X) == 0 and in_matrix.dot(X) == in_pointArray
    nPoints = in_pointArray.shape[1]
    A = np.concatenate((in_matrix, in_plane.reshape((1, 4))))
    right = np.concatenate((in_pointArray, np.zeros((1, nPoints))))
    return np.linalg.solve(A, right)
