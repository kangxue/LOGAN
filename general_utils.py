from scipy.spatial.transform import Rotation
import math
import numpy as np
from numpy.linalg import norm


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
					 

ViewProjectionMatrix = np.matmul(  rotation_matrix([0,0,1],  -110*3.14/180),  rotation_matrix([1,0,0],  -60*3.14/180)   )
				 
def plot_3d_point_cloud_to_Image(x, y, z, dataIs2D=False ):

    img = np.ones( (250,250), np.uint8) * 255
	
    if dataIs2D:
        x2 = y
        y2 = x
    else:
        #x2, y2, _ = proj3d.proj_transform( x, y, z, ViewProjectionMatrix )
        xyz = np.matmul( np.column_stack( (x, y, z) ), ViewProjectionMatrix )
        x2 = xyz[:,0]
        y2 = xyz[:,1]
	
	
    x2 = ( (x2 + 0.5) * 250   ).astype(int)  
    y2 = ( (y2 + 0.5) * 250  ).astype(int) 
	
    x2[ x2 > 249 ] = 249
    y2[ y2 > 249 ] = 249
	
    x2[ x2 < 0 ] = 0
    y2[ y2 < 0 ] = 0
	
    for i in range( x2.shape[0] ):
        img[y2[i], x2[i]] = 0
	
    return img
