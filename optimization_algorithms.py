import numpy as np
from quadprog import solve_qp
from support import wsGram, scale, cv_split, bigGram, solve_singular

def sspline(Gram1, Gram2, y, mscale, lambda_0):
    '''
    Fits a smooth spline to the system defined by X. Gram1 and Gram2 should be the
    same reproducing kernel, defined by the features of X

    Parameters
    ----------
    Gram1 : 3D array
        The first Gram matrix.
    Gram2 : 3D array
        The second Gram matrix (should be the same).
    y : 1D array
        The output y.
    mscale : 1D array
        The weights assigned to the functional components.
    lambda_0 : float
        The regularization parameter.

    Returns
    -------
    dict
        A dictionary with keys b_hat, c_hat.

    '''
    n = len(y)
    
    R_theta1 = wsGram(Gram1, mscale)
    R_theta2 = wsGram(Gram2, mscale)
    
    LHS = np.matmul(R_theta1.T, scale(R_theta1)) + 2*n*lambda_0*R_theta2
    RHS = np.matmul(R_theta1.T, (y-np.mean(y)))
    
    chat = solve_singular(LHS,RHS)
    bhat = np.mean(y-np.matmul(R_theta1,chat))
    return {'b_hat':bhat,
            'c_hat':chat}

def cvlam_Gaussian(Gram1, Gram2, y, mscale, n_folds, lambda_cand=None):
    '''
    Performs cross-validation to find an optimum value for lambda
    using spline interpolation

    Parameters
    ----------
    Gram1 : 3D array
        First reproducing kernel.
    Gram2 : 3D array
        Second reproducing kernel.
    y : 1D array
        output Y.
    mscale : 1D array or list
        The weights assigned to the functional components.
    n_folds : int
        The number of folds the cross-validation should be performed over.
    lambda_cand : 1D array or list, optional
        Candidate lambdas. The default is 2**(np.arange(-10,-22,-0.75)). This is
        not a very good range, needs to change

    Returns
    -------
    float
        The best lambda from cross_validation.

    '''
    if lambda_cand is None:
        lambda_cand = 2**(np.arange(-10,-22,-0.75))
    
    n = len(y)

    R_theta1 = wsGram(Gram1, mscale)
    R_theta2 = wsGram(Gram2, mscale)
    
    kfold = cv_split(n, nfolds = n_folds)
    cv_raw = np.zeros((n,len(lambda_cand)))
    
    for train_index, test_index in kfold:
        train_R_t1 = R_theta1[train_index,:]
        test_R_t1 = R_theta1[test_index,:]
        LHS = np.matmul(train_R_t1.T, train_R_t1)
        RHS = np.matmul(train_R_t1.T, (y[train_index] - np.mean(y[train_index])))
        Cs = np.zeros((n,len(lambda_cand)))
        for k in range(len(lambda_cand)):
            LHS_2 = LHS + 2*len(train_index)*lambda_cand[k]*R_theta2
            chat = solve_singular(LHS_2, RHS)
            bhat = np.mean(y[train_index] - np.matmul(train_R_t1,chat))
            fpred = bhat + np.matmul(test_R_t1, chat)
            cv_raw[test_index, k] = ((fpred - y[test_index])**2).reshape(-1,)
    cvm = np.mean(cv_raw, axis=0)
    optLambd = np.where(cvm == np.min(cvm))
    return lambda_cand[optLambd]


def cvlam_Gaussian_eff(Gram1, Gram2, y, mscale, n_folds, lambda_cand=None):
    '''
    The hope was to somehow vectorize the solver. 
    The rest of the function  works fine, but I need to figure out how to do that.
    Until then, this function is slower than the original

    Parameters
    ----------
    Gram1 : 3D array
        First reproducing kernel.
    Gram2 : 3D array
        Second reproducing kernel.
    y : 1D array
        output Y.
    mscale : 1D array or list
        The weights assigned to the functional components.
    n_folds : int
        The number of folds the cross-validation should be performed over.
    lambda_cand : 1D array or list, optional
        Candidate lambdas. The default is 2**(np.arange(-10,-22,-0.75)). This is
        not a very good range, needs to change

    Returns
    -------
    float
        The best lambda from cross_validation.

    '''
    if lambda_cand is None:
        lambda_cand = 2**(np.arange(-10,-22,-0.75))
    
    n = len(y)

    R_theta1 = wsGram(Gram1, mscale)
    R_theta2 = wsGram(Gram2, mscale)
    # Repeating this lambda_cand times to do cross-val
    R_theta_2_rep = np.repeat(R_theta2[:,:,np.newaxis],len(lambda_cand),axis=2)
    
    kfold = cv_split(n, nfolds = n_folds)
    cv_raw = np.zeros((n,len(lambda_cand)))
    
    for train_index, test_index in kfold:
        train_R_t1 = R_theta1[train_index,:]
        test_R_t1 = R_theta1[test_index,:]
        LHS = np.matmul(train_R_t1.T, train_R_t1)
        RHS = np.matmul(train_R_t1.T, (y[train_index] - np.mean(y[train_index])))
        
        LHS_rep = np.repeat(LHS[:,:,np.newaxis],len(lambda_cand),axis=2)
        LHS_2_rep = R_theta_2_rep * lambda_cand * len(train_index)*2 + LHS_rep
        LHS_2_rep = LHS_2_rep.transpose((2,0,1))
        c_hat = np.array([])
        for slice in LHS_2_rep:
            L = np.linalg.cholesky(slice)
            c = np.linalg.solve(L.T,np.linalg.solve(L,RHS)).reshape(-1,)
            c_hat = np.append(c_hat,c)
            
        c_hat = c_hat.reshape(n,LHS_2_rep.shape[0],order='F')
        b_hat = np.mean(y[train_index] - np.matmul(train_R_t1,c_hat),axis=0)
        fpred = np.matmul(test_R_t1,c_hat) + b_hat
        cv_raw[test_index,:] = (fpred - y[test_index])**2

    cvm = np.mean(cv_raw, axis=0)
    optLambd = np.where(cvm == np.min(cvm))
    return lambda_cand[optLambd]


 
def SSANOVAwt_Gaussian(Gram1,Gram2,y,mscale, order,cat_pos):
    '''
    Generates the initial adaptive weights w using SS_ANOVA

    Parameters
    ----------
    x : 2D array
        input array X.
    y : 1D array
        response vector X.
    mscale : list
        functional component weights.
    nbasis : integer, optional
        The number of knots. The default is None,
        If None is passed, the algorithm will select a number

    Returns
    -------
    1D vector
        The adaptive weights w_0.

    '''
    #n = len(x)
    d = len(mscale)
    lam = cvlam_Gaussian(Gram1,Gram2,y,mscale,8)
    cc = sspline(Gram1, Gram2, y, mscale, lam)
    c_hat = cc['c_hat']
    L2 = np.zeros(d)
    Gram1 = np.swapaxes(Gram1, 0, 2)
    Gram1 = np.swapaxes(Gram1, 2, 1)
    fjj = np.matmul(Gram1, c_hat).T
    #print(fjj.shape)
    #fjj = fjj[0,:,:]
    fjj = fjj.round(9)
    lengths = np.array([len(np.unique(fjj[:,j])) for j in range(d)])

    idx1 = np.where(lengths > 6)[0]
    idx2 = np.where(lengths <= 6)[0]
    
    L2[idx1] = np.sqrt(np.mean(fjj[:,idx1]**2,axis=0))
    rng = np.max(fjj[:,idx2],axis=0) - np.min(fjj[:,idx2],axis=0)
    L2[idx2] = rng

            
    return 1/L2

def twostep_Gaussian(Gram1, Gram2, y, wt, lam, mm, eps = 1e-3,maxit=20):
    '''
    Solves the system that minimizes ||z-G*theta|| using the iterative algo in
    the COSSO paper (more in README)

    Parameters
    ----------
    Gram1 : 3D array
        First Gram matrix.
    Gram2 : 3D array
        Second Gram matrix. if full basis is desired, Gram1 and Gram2 will be 
        the same array
    y : 1D array
        Output vector y.
    wt : list
        Adaptive weights w.
    lam : float
        Regularization penalty lambda.
    mm : float
        Constraint for M.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    '''
    n = len(y)
    nbasis = Gram1.shape[1]
    d = len(wt)
    wt = np.array(wt,dtype=float)
    theta = np.ones(d)
    # Solves iteratively
    iteration = 0
    while True:
        if iteration > maxit:
            break
        iteration +=1
        cb0 = sspline(Gram1, Gram2, y, theta/(wt**2), lam)
        c0 = cb0['c_hat']
        b0 = cb0['b_hat']
        G1 = np.zeros((n,d))
        G2 = np.zeros((nbasis,d))
        for j in range(d):
            G1[:,j] = (np.matmul(Gram1[:,:,j], c0)*(wt[j]**-2)).reshape(-1,)
            G2[:,j] = (np.matmul(Gram2[:,:,j], c0)*(wt[j]**-2)).reshape(-1,)
        dvec = (np.matmul(2*G1.T,y-b0)-np.matmul(n*lam*G2.T,c0)).reshape(-1,)
        Dmat = 2*np.matmul(G1.T,G1)
        Amat = (np.vstack([np.diag(np.ones(d)),np.array([-1]*d)])).T
        bvec = np.hstack([np.zeros((d)),-mm])
        obj = solve_qp(Dmat,dvec,Amat,bvec)
        theta_new = obj[0]
        # Stopping criterion
        div = theta_new
        div[theta_new<1e-6] = 1
        rel_norm = np.sqrt(np.sum(((theta - theta_new)/div)**2))  
        if rel_norm < eps:
            theta = theta_new
            break
        theta = theta_new
        
        theta[theta<1e-8] = 0
        
    cb1 = sspline(Gram1, Gram2, y, theta/(wt**2),lam)
    output = {'coefs':cb1['c_hat'],
              'intercept':cb1['b_hat'],
              'theta':theta
              }
    return output

def cosso_Gaussian(Gram, y, wt, basis_idx):
    '''
    Builds a cosso dict, which contains:
        1. basis_idx, the index of the knots from X
        2. a dictionary which contains:
            2.1 opt_lam - the best lambda found via cross-val
            2.2 Mgrid - the range of M vals to be tuned
            2.3 L2norm - the L2 norms associated with each value of M (row-wise)
                        columns correspond to the features of X

    Parameters
    ----------
    Gram : 3D array
        The Gram matrix of reproducing kernels corresponding to X.
    y : 1D array
        The outut vector y.
    wt : list
        The adaptive weights w.
    nbasis : int, optional
        The number of knots used to build the acosso splines. The default is None, and the algo decides

    Returns
    -------
    dict
        A dictionary, containing cosso data to be tuned.

    '''
    #n = len(y)
    p = len(wt)
    Gramat1 = Gram[:,basis_idx,:]
    Gramat2 = Gramat1[basis_idx,:,:]
    wt = np.array(wt,dtype=float)
    bestlam = cvlam_Gaussian(Gramat1, Gramat2, y, 1/(wt**2), n_folds = 6)
    L2normMat = np.array([0]*p)
    Mgrid = [0]
    tempM = 0.2
    tempTheta = np.zeros(p)
    tempL2norm = np.zeros(p)
    loop = 0
    while (((np.sum(tempTheta)>1e-7)<p) & 
           (loop <= [np.floor(2*p) if p<=15 else p][0])):
        loop += 1
        Mgrid.append(tempM)
        tmpSol = twostep_Gaussian(Gramat1, Gramat2, y, wt, bestlam, tempM)
        temp_theta = tmpSol['theta']
        temp_coefs = tmpSol['coefs']
        for j in range(p):
            temp_L2normMat = np.sqrt(np.mean((temp_theta[j]/wt[j]**2 *\
                                           np.matmul(Gramat1[:,:,j],temp_coefs))**2))
            L2normMat = np.append(L2normMat, temp_L2normMat)
            if len(np.unique(Gramat1[:,:,j].round(9)))>6:
                tempL2norm[j] = temp_L2normMat
            else:
                up = np.max(temp_theta[j]/wt[j]**2 * np.matmul(Gramat1[:,:,j],temp_coefs))
                lo = np.min(temp_theta[j]/wt[j]**2 * np.matmul(Gramat1[:,:,j],temp_coefs))
                tempL2norm[j] = up-lo
        
        if loop < 10:
            tempM += 0.25
        elif loop >= 10 and loop < 16:
            tempM += 0.5
        elif loop >=16 and loop < 20:
            tempM += 1
        else:
            tempM += 2
    
    L2normMat = L2normMat.reshape(loop+1,p)
    
    return {'tune': {'opt_lam':bestlam,
                     'Mgrid':Mgrid,
                     'L2norm':L2normMat}
            }


def tune_cosso_Gaussian(cosso_dict, n_folds):
    n = len(cosso_dict['y'])
    p = len(cosso_dict['wt'])
    
    bound = p//2 if p<=15 else p//3
    
    origMgrid = np.array(cosso_dict['tune']['Mgrid'][1:])
    L2 = cosso_dict['tune']['L2norm'][1:]
    uniqueSize = np.unique(np.sum((L2>0),axis=1))
    mask = np.sum((L2>0),axis=1) <= bound
    newGrid = origMgrid[mask]
    
    for k in range(len(uniqueSize)):
        if uniqueSize[k] > bound:
            new_val = origMgrid[np.max(np.where(np.sum((L2>0),axis=1)==uniqueSize[k]))]
            newGrid = np.append(newGrid, new_val)

    newGrid = np.append(np.array([0]), np.sort(newGrid))
    uniqueSize = np.append(np.array([0]), np.sort(uniqueSize))
    
    refinePt = np.where((uniqueSize[1:] - uniqueSize[:-1]) > 1)[0]
    
    if len(refinePt)>0:
        refinePt1 = refinePt[refinePt<10]
        l1 = np.percentile(np.hstack([origMgrid[refinePt1],origMgrid[refinePt1+1]]),[0.3,0.6])
        
        refinePt2 = refinePt[refinePt>=10]
        l2 = np.mean(np.hstack([origMgrid[refinePt2],origMgrid[refinePt2+1]]))
        if np.isnan(l2):
            l2 = np.array([])
        extMgrid = np.hstack((l1,l2))
    else:
        extMgrid = np.array([])
        
    kfold = cv_split(n, nfolds = n_folds)
    cand_M = np.sort(np.hstack((newGrid[1:], extMgrid)))
    cv_raw = np.zeros((n,len(cand_M)))
    
    for train_index, test_index in kfold:
        trainGramat1 = cosso_dict['Kmat'][train_index,:,:]
        trainGramat1 = trainGramat1[:,cosso_dict['basis_idx'],:]
        testGramat1 = cosso_dict['Kmat'][test_index,:,:]
        testGramat1 = testGramat1[:,cosso_dict['basis_idx'],:]
        Gramat2 = cosso_dict['Kmat'][cosso_dict['basis_idx'],:,:]
        Gramat2 = Gramat2[:,cosso_dict['basis_idx'],:]
        for m in range(len(cand_M)):
            tempSol = twostep_Gaussian(trainGramat1, Gramat2,
                                       cosso_dict['y'][train_index],
                                       cosso_dict['wt'],
                                       cosso_dict['tune']['opt_lam'],
                                       cand_M[m])
            ws_temp = wsGram(testGramat1,tempSol['theta']/cosso_dict['wt']**2)
            tempfpred = tempSol['intercept'] + np.matmul(ws_temp,tempSol['coefs'])
            cv_raw[test_index,m] = ((cosso_dict['y'][test_index]-tempfpred)**2).reshape(-1,)
        
        cvm = np.mean(cv_raw,axis=0)
        cvsd = np.sqrt(np.mean((cv_raw - cvm)**2, axis=0) / n)
        optM = cand_M[np.where(cvsd == np.min(cvsd))]

    return {'OptM':optM,
            'M':cand_M,
            'cvm':cvm,
            'cvsd':cvsd
            }
    
