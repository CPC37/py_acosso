import numpy as np
import numpy as np
from math import factorial
from support import bigGram, wsGram
from optimization_algorithms import SSANOVAwt_Gaussian,twostep_Gaussian, \
    cosso_Gaussian, tune_cosso_Gaussian
from random import sample


class acosso:
    def __init__(self, X, y, nbasis=None, basis_idx=None, order = 2, wt=None,
                 gamma=2, cat_pos = None, m_scale0 = None):
        self.X = X.copy()
        self.rescale = False
        self.fix_scale()
        if self.rescale == True:
            print('Warning: X matrix was rescaled')
            
        self.n = X.shape[0]
        self.d = X.shape[1]
        
        self.y = y.copy()
        
        if (nbasis is None) and (basis_idx is None):
            self.nbasis = X.shape[0]
            self.basis_idx = np.sort(sample(range(self.n),self.nbasis))
        elif (nbasis is not None) and (basis_idx is None):
            self.nbasis = nbasis     
            self.basis_idx = np.sort(sample(range(self.n),self.nbasis))
        else:
            self.basis_idx = basis_idx
            
        self.gamma = gamma 
        self.m_scale0 = m_scale0
        
        
        if cat_pos is None:
            self.cat_pos_fix()
        else:
            self.cat_pos = cat_pos
            
        self.order = order
        

        if cat_pos is None:
            cat_pos = []
            
    def choose_P(self,order):
        '''
        A function to generate the number of possible interaction terms

        Parameters
        ----------
        order : integer
            the order of the interactions.

        Returns
        -------
        P : integer
            the number of terms.

        '''
        d = self.X.shape[1]
        if order == 1:
            P = d
        elif order ==2 :
            P = d+ int(d*(d-1)/2)
        elif order == 3:
            P = d + int(d*(d-1)/2) + int(factorial(d)/factorial(d-3)/factorial(3))
        return P
    

    
    def cat_pos_fix(self):
        '''
        If the categorical variable position is not specified, this looks for 
        features with less than 7 distinct values and assumes them to be categorical

        Returns
        -------
        None.

        '''
        d = self.X.shape[1]
        if not self.cat_pos:
            self.cat_pos = []
            for j in range(d):
                if len(np.unique(self.X[:,j])) < 7:
                    self.cat_pos = self.cat_pos + [j]
        
    def fix_scale(self):
        '''
        Looks at whether X is scaled between [0,1]. If not, X is re-scaled

        Returns
        -------
        None.

        '''
        d = self.X.shape[1]
        for j in range(d):
            if ((np.max(self.X[:,j])>1) | (np.min(self.X[:,j])<0)):
                self.X[:,j] = (self.X[:,j] - np.min(self.X[:,j])) / (np.max(self.X[:,j])-np.min(self.X[:,j]))
                self.rescale = True
                
    
    # def get_knots(self):
    #     '''
    #     Gets the number of specified knots from the data. If nbasis is not specified,
    #     the algorithm chooses on its' own, ensuring that a fair number of chosen from each category

    #     Returns
    #     -------
    #     DONOTUSE - work in progress.

    #     '''
    #     levels = np.unique(self.X[:,self.cat_pos],axis=0)
    #     #levels = [(i,j) for i in self.cat_pos for j in self.X[:,i]]
    #     n_levels = levels.shape[0]
        
    #     basis_idx = np.array([])
        
    #     # The numbere of points we need to sample for each cat level
    #     nbasis_level = self.nbasis // n_levels
    #     rem = self.nbasis % nbasis_level
        
    #     for level in levels:
    #         pool = np.where(self.X[:,col]==level)
    #         # Try to pick them. If this fails, the number of basis points is too large
    #         try:
    #             basis_idx = np.append(basis_idx,np.sort(sample(pool, nbasis_level)))
                
    #         except ValueError:
    #             print('''nbasis is too high''')
                
    #     rem_idx = [i for i in range(self.n) if i not in basis_idx]
    #     basis_idx = np.append(basis_idx, np.sort(sample(rem_idx, rem)))
    #     self.basis_idx = basis_idx
    def build(self):
        '''
        Builds the Gram matrices of the system. This can be quite intensive,
        especially if the input has a large number of rows

        Returns
        -------
        None.

        '''
        
        self.Kmat = bigGram(self.X, self.X, order = self.order,
                            cat_pos = self.cat_pos)
        
        self.P = self.Kmat.shape[2]
        
        self.Gramat1 = self.Kmat[:,self.basis_idx, :]
        
        self.Gramat2 = self.Gramat1[self.basis_idx,:,:]
        
        self.wt = self.SSANOVAwt(mscale = self.m_scale0, gamma = self.gamma)
        
        

    def SSANOVAwt(self, mscale = None,gamma=2):
        '''
        Function call to calculate adaptive weights wt. 
        MEMORY WARNING: this separately builds the gram matrices

        Parameters
        ----------
        mscale : 1D array, optional
            Not necessary. The default is None, which defaults to [1,1,1...]
        gamma : int, optional
            The power to which we should raise the norm. The default is 2.

        Returns
        -------
        1D array
            The adaptive weights wt.

        '''
        if mscale is None:
            mscale = np.ones(self.P)
        inv_l2norm = SSANOVAwt_Gaussian(self.Gramat1,self.Gramat2,
                                        self.y,mscale,order=self.order,
                                        cat_pos = self.cat_pos)
        return inv_l2norm**gamma
    
    
    def fit(self):
        '''
        Finds the best lambda for the norm penalty

        Returns
        -------
        None. Creates an obj dictionary as an attribute of acosso

        '''
        self.obj = cosso_Gaussian(self.Kmat, self.y, self.wt, self.basis_idx)
        self.obj['y']=self.y
        self.obj['X']=self.X
        self.obj['Kmat'] = self.Kmat
        self.obj['basis_idx'] = self.basis_idx
        self.obj['wt'] = self.wt
    
    def tune(self, n_folds=5):
        '''
        The function to tune M

        Returns
        -------
        None, Opt M added to obj attribute
            floating point number - best M.

        '''

        obj_tuning = tune_cosso_Gaussian(self.obj,n_folds)
        self.obj['OptM'] = obj_tuning['OptM']
        self.obj['M'] = obj_tuning['M']
        self.obj['Mcvm'] = obj_tuning['cvm']
        self.obj['Mcvsd'] = obj_tuning['cvsd']
        fit_func = twostep_Gaussian(self.Gramat1,
                            self.Gramat2,
                            self.y,
                            self.wt,
                            self.obj['tune']['opt_lam'],
                            self.obj['OptM']
                            )
        self.obj['coefs']=fit_func['coefs']
        self.obj['intercept'] = fit_func['intercept']
        self.obj['theta'] = fit_func['theta']
    
    def predict(self, x_new, eps = 1e-7):
        d = x_new.shape[1]
        for i in range(d):
            if (np.any((x_new[:,i] > 1)) | np.any((x_new[:,i]<0))):
                x_new[:,i] = (x_new[:,i] - np.min(x_new[:,i])) /\
                    (np.max(x_new[:,i])- np.min(x_new[:,i]))
                print(f'Warning! Column {i} was rescaled')
        Gramat1 = self.Gramat1
        Gramat2 = self.Gramat2
        fit_obj = twostep_Gaussian(Gramat1,
                                  Gramat2,
                                  self.obj['y'],
                                  self.obj['wt'],
                                  self.obj['tune']['opt_lam'],
                                  self.obj['OptM'][0])
        
        BG = bigGram(x_new, 
                     self.obj['X'][self.obj['basis_idx'],:],
                     order = self.order,
                     cat_pos = self.cat_pos)
        
        mscale = fit_obj['theta']/(self.obj['wt'])**2
        predictor = fit_obj['intercept'] + np.matmul(wsGram(BG,mscale),fit_obj['coefs'])
        return predictor
