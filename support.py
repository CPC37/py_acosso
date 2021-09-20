import numpy as np
from math import factorial
from sklearn.model_selection import KFold
from scipy.spatial import distance_matrix



def genK(x1,x2,kern_type):
    if kern_type == 'Gaussian':
        K = distance_matrix(x1.reshape(len(x1),1),x2.reshape(len(x2),1))
        K = np.exp(-K**2)
        
    elif kern_type == 'RKHS':
        m = len(x1)
        n = len(x2)
    
        x1 = x1.reshape(m,1)
        x2 = x2.reshape(n,1)
    
        b = abs(np.matmul(x1,np.ones((1,n))) - np.matmul(np.ones((m,1)),x2.T))
        
        k1s = x1 - 0.5
        k1t = x2 - 0.5
        k2s = (k1s**2 - 1/12)/2
        k2t = (k1t**2 - 1/12)/2
    
        K = (np.matmul(k1s,k1t.T) + np.matmul(k2s, k2t.T) - ((b-0.5)**4 - (b-0.5)**2 / 2 + 7/240)/24)
        
    elif kern_type == 'Cate':
        n1 = len(x1)
        n2 = len(x2)
    
        # We need to reshape these to ensure they are n x 1 arrays (seriously, to hell with numpy)
        x1 = x1.reshape(n1,1)
        x2 = x2.reshape(n2,1)
     
        t1 = np.tile(x1, [n2,1])
        t2 = np.repeat(x2, n1)
        # Having to reshape for the same reason as above
        t2 = t2.reshape(len(t2),1)
    
        L = len(np.unique(np.concatenate([t1, t2])))
        K = (L*(t1==t2) - 1).reshape((n1,n2),order='F')
    
        K = K/L
        
    return K





def bigGram(x1, x2, order, cat_pos=0, kern_types = None):
    '''Creates a Gram matrix of the entire space of x1 and x2. Usually this function is used with x1==x2
    The output of this function is a 3D matrix containing the kernel of each column

    Parameters
    ----------
    x1 : array
        array numero uno.
    x2 : array
        array numero dos.
    order : int
        The order of interactions
    cat_pos : array or numeric
        The location of the categorical column(s)

    Returns
    -------
    Gram : array
        the gram matrix - a 3D numpy array, the 3rd dimension is indicative of the k matrices

    '''

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    d = x1.shape[1]
    
    if kern_types is None:
        kern_types = np.array(['RKHS']*d)
    
    Gram = np.zeros((n1,n2,d))
        
    if type(cat_pos) != list:
        cat_pos = [cat_pos]
    
    kern_types[cat_pos] = '''Cate'''
    
    for j in range(d):
        Gram[:,:,j] = genK(x1[:,j],x2[:,j], kern_types[j])
        
    
    if order == 2:
        combs = int(d*(d-1)/2)
        Gram_2 = np.zeros((n1,n2,d+combs))
        Gram_2[:,:,0:d] = Gram
        next_ind = d
        for i in range(d-1):
            for j in range(i+1,d):
                gram_term = Gram[:,:,i] * Gram[:,:,j]
                Gram_2[:,:,next_ind] = gram_term
                next_ind += 1
        Gram = Gram_2
        
    if order == 3:
        # combs1 and combs2 are realisations of n!/((n-r)!r!)
        # This is combinations of d taken as 2
        combs1 = int(d*(d-1)/2)
        # This is combinations of d taken as 3
        combs2 = int(factorial(d)/factorial(d-3)/factorial(3))
        
        # Adding first order terms
        Gram_3 = np.zeros((n1,n2,d+combs1+combs2))
        Gram_3[:,:,0:d] = Gram
        # Adding second order terms
        next_ind = d
        for i in range(d-1):
            for j in range(i+1,d):
                gram_term = Gram[:,:,i] * Gram[:,:,j]
                Gram_3[:,:,next_ind] = gram_term
                next_ind += 1
        # Adding third order terms
        for i in range(d-2):
            for j in range(i+1,d-1):
                for k in range(j+1,d):
                    gram_term = Gram[:,:,i] * Gram[:,:,j] * Gram[:,:,k]
                    Gram_3[:,:,next_ind] = gram_term
                    next_ind += 1
        Gram = Gram_3
    return Gram

def wsGram(Gram, mscale):
    '''
    Creates the array K_theta, the addition of the Gram matrices multiplied by their scale parameters.
    mscale = theta * w^-2

    Parameters
    ----------
    Gram : 3D array
        The Gram matrix containing the reproducing kernels.
    mscale : = theta*w^-2
        The scaling parameter.

    Returns
    -------
    K_theta : 2D array
        Search for K_theta in the cosso equation.

    '''
    n1 = Gram.shape[0]
    n2 = Gram.shape[1]
    d = Gram.shape[2]
    K_theta = np.zeros((n1,n2))
    for j in range(d):
        K_theta = K_theta + mscale[j]*Gram[:,:,j]
    return K_theta

def solve_singular(A,b):
    '''
    Adds a small nugget to the matrix A to allow for an approximate solution
    where otherwise it would fail

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    obj : TYPE
        DESCRIPTION.

    '''
    a = A.copy()
    try:
        obj = np.linalg.solve(a,b)
    except:
        a = a + 1e-7*np.diag(np.ones(a.shape[0]))
        obj = np.linalg.solve(a,b)
    return obj

def scale(X):
    '''
    Scales by subtracting the mean from the columns. It only serves to add a 
    slight nugget to the terms in X

    Parameters
    ----------
    X : 2D array
        Input.

    Returns
    -------
    x : 2D array
        scaled output.

    '''
    d = X.shape[1]
    x = X.copy()
    for j in range(d):
        x[:,j] = x[:,j] - np.mean(x[:,j])
    return x


def cv_split(n, nfolds = 5, seed = 19690731):
    '''
    Generates the k-fold cross validation indices using sklearn.metric_selection.KFold

    Parameters
    ----------
    n : int
        The length of a matrix for which CV is required.
    nfolds : int, optional
        The number of folds. The default is 5.
    seed : numeric, optional
        The random seed used in sklearn kf. The default is 19690731.

    Returns
    -------
    kf : KFold object
        The folds contained in a KFold object.

    '''
    X = np.zeros((n,1))
    kf = KFold(n_splits = nfolds, shuffle = True, random_state = seed)
    return kf.split(X)
