"""
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of the multiline TRL calibration algorithm with linear 
uncertainty propagation capabilities. This is an extension of my original 
mTRL algorithm [1] to account for uncertainties with the help of 
METAS UncLib [2]. Summery details can be found here: https://arxiv.org/abs/2206.10209

Developed at:
    Institute of Microwave and Photonic Engineering,
    Graz University of Technology (TU Graz), Austria

[1] Z. Hatab, M. Gadringer, and W. B¨osch, 
"Improving the reliability of the multiline trl calibration algorithm," 
in 98th ARFTG Microwave Measurement Conference, Las Vegas, NV, USA, 2022.

[2] M. Zeier, J. Hoffmann, and M. Wollensack, 
"Metas.UncLib—a measurement uncertainty calculator for advanced problems," 
Metrologia, vol. 49, no. 6, pp. 809–815, nov 2012.
"""

import numpy as np
import skrf as rf
import metas_unclib as munc
munc.use_linprop()

c0 = 299792458   # speed of light in vacuum (m/s)

def metas_or_numpy_funcs(metas=False):
    # this is my way to switch between metas and numpy functions
    if metas:
        dot   = munc.ulinalg.dot
        inv   = munc.ulinalg.inv
        eig   = munc.ulinalg.eig
        solve = munc.ulinalg.solve
        conj  = munc.umath.conj
        exp   = munc.umath.exp
        log   = munc.umath.log
        acosh = munc.umath.acosh
        sqrt  = munc.umath.sqrt
        get_value = munc.get_value
        ucomplex  = munc.ucomplex
    else:
        dot   = np.dot
        inv   = np.linalg.inv
        eig   = np.linalg.eig
        solve = np.linalg.solve
        conj  = np.conj
        exp   = np.exp
        log   = np.log
        acosh = np.arccosh
        sqrt  = np.sqrt
        get_value = lambda x: x
        ucomplex  = complex
    return dot, inv, eig, solve, conj, exp, log, acosh, sqrt, get_value, ucomplex

def correct_switch_term(S, G21, G12):
    # correct switch terms of measured S-parameters at a single frequency point
    # G21: forward (sourced by port-1)
    # G12: reverse (sourced by port-2)
    S_new = S.copy()
    S_new[0,0] = (S[0,0]-S[0,1]*S[1,0]*G21)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[0,1] = (S[0,1]-S[0,0]*S[0,1]*G12)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[1,0] = (S[1,0]-S[1,1]*S[1,0]*G21)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[1,1] = (S[1,1]-S[0,1]*S[1,0]*G12)/(1-S[0,1]*S[1,0]*G21*G12)
    return S_new

def S2T(S):
    # convert S- to T-parameters at a single frequency point
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T/S[1,0]

def T2S(T):
    # convert T- to S-parameters at a single frequency point
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S/T[1,1]

def compute_G_with_takagi(A, metas=False):
    # implementation of Takagi decomposition to compute the matrix G used to determine the weighting matrix.
    #
    # Singular value decomposition for the Takagi factorization of symmetric matrices
    # https://www.sciencedirect.com/science/article/pii/S0096300314002239
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=metas)
    if metas:
        u,s = eig(A.dot(conj(A).T))
    else:
        s,u = eig(A.dot(conj(A).T))
    s = sqrt(s).real  # singular values. They need to come as real positive floats
    s = s*np.sign(get_value(s))
    inx = np.flip(np.argsort(get_value(s))) # sort in increasing order
    lambd = s[inx][0]*s[inx][1]             # this is the eigenvalue of the calibration eigenvalue problem
    u = u[:,inx][:,:2]  # low-rank truncated (Eckart-Young-Mirsky theorem)
    A = (A + A.T)/2  # only consider the symmetric part
    phi = sqrt( conj(np.diag(u.T.dot(conj(A)).dot(u))) )
    G = u.dot(np.diag(phi))
    return G, lambd

def WLS(x,y,w=1, metas=False):
    # Weighted least-squares for a single parameter estimation
    # determine gamma after the calibration coefficients are solved
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=metas)
            
    x = x*(1+0j) # force x to be complex type 
    # return (x.conj().transpose().dot(w).dot(y))/(x.conj().transpose().dot(w).dot(x))
    return dot(x.conj().transpose().dot(w), y)/dot(x.conj().transpose().dot(w), x)

def Vgl(N):
    # inverse covariance matrix for propagation constant computation
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones(shape=(N-1, N-1), dtype=complex)

def compute_gamma(X, M, lengths, gamma_est, metas=False):
    # determine gamma after normalized calibration coefficients are solved
    # gamma is determined through linear weighted least-squares
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=metas)
    
    EX = dot(inv(X), M)[[0,-1],:]     # extract z and y columns
    EX = np.diag(1/EX[:,0]).dot(EX)   # normalize to the thru line
    gamma_l = log((EX[0,:] + 1/EX[-1,:])/2)
    gamma_l = gamma_l[get_value(lengths) != 0]
    l = -lengths[get_value(lengths) != 0]
    n = np.round(get_value(gamma_l - gamma_est*l).imag/np.pi/2)
    gamma_l = gamma_l - 1j*2*np.pi*n # unwrapped
    
    return WLS(l, gamma_l, Vgl(len(l)+1))


def mTRL_at_one_freq(Slines, lengths, Sreflect, ereff_est, reflect_est_a, reflect_est_b, f, sw=[0,0]):
    # Performing a standard mTRL without uncertainty. That is, METAS package not used.
    # Slines: array containing 2x2 S-parameters of each line standard
    # lengths: array containing the lengths of the lines
    # Sreflect: 2x2 S-parameters of the measured reflect standard
    # ereff_est: complex scalar of estimated effective permittivity
    # reflect_est: complex scalar of estimated reflection coefficient of the reflect standard
    # f: scalar, current frequency point
    # sw: 1x2 array holding the forward and reverse switch terms, respectively.
    
    # numpy functions
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=False)
    
    reflect_est_b = reflect_est_a if reflect_est_b is None else reflect_est_b
    
    # correct switch term
    Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sw) else Slines
    Sreflect = correct_switch_term(Sreflect,sw[0],sw[1]) if np.any(sw) else Sreflect # this is actually not needed!
    
    # make first line as Thru, i.e., zero length
    lengths = np.array([x-lengths[0] for x in lengths])
    
    # measurements
    Mi    = [S2T(x) for x in Slines] # convert to T-parameters        
    M     = np.array([x.flatten('F') for x in Mi]).T
    MinvT = np.array([inv(x).flatten('F') for x in Mi])
          
    ## Compute W from Takagi factorization
    G, lambd = compute_G_with_takagi(MinvT.dot(M[[0,2,1,3]]), metas=False)
    W = conj((G@np.array([[0,1j],[-1j,0]])).dot(G.T))
    
    # estimated gamma to be used to resolve the sign of W
    gamma_est = 2*np.pi*f/c0*np.sqrt(-(ereff_est-1j*np.finfo(float).eps))  # the eps is to ensure positive square-root
    y_est = np.exp(gamma_est*get_value(lengths))
    z_est = 1/y_est
    if np.sign((y_est.dot(get_value(W)).dot(z_est)).real)-1:
        W = -W # resolve the sign ambiguity
    
    ## Solving the weighted eigenvalue problem
    F = M.dot(W).dot(MinvT[:,[0,2,1,3]]) # weighted measurements
    eigval, eigvec = eig(F) # numpy order
    inx = np.argsort(get_value(eigval.real)) # get indices of sorted eigenvalues
    lambd = (eigval[inx[-1]] - eigval[inx[0]])/2   # lambda for debugging; over ride lambd from G (both should be the same)
    
    x1_ = eigvec[:,inx[0]]
    x4  = eigvec[:,inx[-1]]
    x1_ = x1_/x1_[0]
    x4  = x4/x4[-1]
    x2_ = np.array([x4[2], ucomplex(1), x4[2]*x1_[2], x1_[2]])
    x3_ = np.array([x4[1], x4[1]*x1_[1], ucomplex(1), x1_[1]])
    
    X_ = np.array([x1_, x2_, x3_, x4]).T  # normalized calibration matrix
    
    ## Compute propagation constant
    gamma = compute_gamma(X_, M, lengths, gamma_est, metas=False)
    gamma = gamma.real*np.sign(get_value(gamma.real)) + 1j*gamma.imag # make sure to get the correct sign for losses
    ereff = -(c0/2/np.pi/f*gamma)**2

    ## De-normalization
    # solve for a11b11 and k from Thru measurement
    ka11b11,_,_,k = solve(X_, M[:,0])
    #ka11b11,_,_,k = inv(X_).dot(M[:,0])
    a11b11  = ka11b11/k
    # solve for a11/b11, a11 and b11
    Ga = Sreflect[0,0]
    Gb = Sreflect[1,1]
    a11_b11 = (reflect_est_b/reflect_est_a)*(Ga - x2_[0])/(1 - Ga*x3_[3])*(1 + Gb*x2_[3])/(Gb + x3_[0])
    a11 = sqrt(a11_b11*a11b11)
    # choose correct answer for a11 and b11
    G0 = get_value( (Ga - x2_[0])/(1 - Ga*x3_[3])/a11 )
    if abs(G0 - get_value(reflect_est_a)) > abs(-G0 - get_value(reflect_est_a)):
        a11 = -a11
    b11 = a11b11/a11
    
    reflect_est_a = (Ga - x2_[0])/(1 - Ga*x3_[3])/a11  # new value
    reflect_est_b = (Gb + x3_[0])/(1 + Gb*x2_[3])/b11  # new value
    
    # build the calibration matrix (de-normalize)
    X  = dot(X_,np.diag([a11b11, b11, a11, ucomplex(1)]) )
    
    return X, k, get_value(ereff), gamma, get_value(reflect_est_a), get_value(reflect_est_b), lambd

def cov_ereff_Gamma(ereff_Gamma, lengths, X, k, f):
    # determine cov of ereff_Gamma (line mismatch)
    
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=True)
    
    ereff = ereff_Gamma[0]
    gamma = 2*np.pi*f/c0*sqrt(-ereff)
    G = ereff_Gamma[1]
    Rkron = np.array([[1/(1 - G**2), G/(1 - G**2), -G/(1 - G**2), -G**2/(1 - G**2)],
                      [G/(1 - G**2), 1/(1 - G**2), -G**2/(1 - G**2), -G/(1 - G**2)],
                      [-G/(1 - G**2), -G**2/(1 - G**2), 1/(1 - G**2), G/(1 - G**2)], 
                      [-G**2/(1 - G**2), -G/(1 - G**2), G/(1 - G**2), 1/(1 - G**2)]])
    
    return np.array( [munc.get_covariance( k*X@dot(Rkron, [exp(-gamma*l),0,0,exp(gamma*l)]) ) for l in lengths] )

def umTRL_at_one_freq(Slines, lengths, Sreflect, ereff_est, reflect_est_a, reflect_est_b, f, X, k, sw=[0,0],
                      uSlines=None, ulengths=None, uSreflect=None, ureflect=None, uereff_Gamma=None, usw=None):
    # Slines: array containing 2x2 S-parameters of each line standard
    # lengths: array containing the lengths of the lines
    # Sreflect: 2x2 S-parameters of the measured reflect standard
    # ereff_est: complex scalar of estimated effective permittivity
    # reflect_est: complex scalar of estimated reflection coefficient of the reflect standard
    # f: scalar, current frequency point
    # sw: 1x2 array holding the forward and reverse switch terms, respectively.
    # X: 4x4 array estimated calibration coefficients
    # k: scalar of estimated 7th term calibration coefficient
    #
    # uSlines: array containing the 8x8 covariance of each line measurement
    # ulengths: array containing the variance of each line
    # uSreflect: 8x8 covariance matrix of measured reflect 
    # ureflect: 2x2 covariance matrix of the reflection coefficient of the reflect standard
    # uereff_Gamma: 4x4 covariance matrix of the line mismatch.
    # usw: 4x4 covariance matrix of switch terms
    
    # metas functions
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=True)
    
    if uSlines is not None:
        Slines   = np.array([munc.ucomplexarray(x, covariance=covx) for x,covx in zip(Slines,uSlines)]) 
    else:
        Slines   = np.array([munc.ucomplexarray(x, covariance=np.zeros((8,8))) for x in Slines]) 
    
    if ulengths is not None:
        lengths  = munc.ufloatarray(lengths, covariance=ulengths)
    else:
        lengths  = munc.ufloatarray(lengths, covariance=np.zeros((len(lengths),len(lengths))))
    
    if uSreflect is not None:
        Sreflect = munc.ucomplexarray(Sreflect, covariance=uSreflect)
    else:
        Sreflect = munc.ucomplexarray(Sreflect, covariance=np.zeros((8,8)))
    
    if ureflect is not None:
        reflect_est_a = munc.ucomplex(reflect_est_a, covariance=ureflect)
        reflect_est_b = munc.ucomplex(reflect_est_b, covariance=ureflect)
    else:
        reflect_est_a = munc.ucomplex(reflect_est_a, covariance=np.zeros((2,2)))
        reflect_est_b = munc.ucomplex(reflect_est_b, covariance=np.zeros((2,2)))
        
    if uereff_Gamma is not None:
        ereff_Gamma = munc.ucomplexarray([ereff_est, 0], covariance=uereff_Gamma)
    else:
        ereff_Gamma = munc.ucomplexarray([ereff_est, 0], covariance=np.zeros((4,4)))
    
    if usw is not None:
        sww = sw # just to use as a check
        sw = munc.ucomplexarray(sw, covariance=usw)
    else:
        sww = sw # just to use as a check
        sw = munc.ucomplexarray(sw, covariance=np.zeros((4,4)))
    
    # correct switch term
    Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sww) else Slines
    Sreflect = correct_switch_term(Sreflect,sw[0],sw[1]) if np.any(sww) else Sreflect # this is actually not needed!
    
    # make first line as Thru, i.e., zero length
    lengths = np.array([x-lengths[0] for x in lengths])
    
    # measurements
    Mi    = [S2T(x) for x in Slines] # convert to T-paramters
    
    # update the covariance of the measurments with impedance mismatch uncertainties
    Mi    = [ munc.ucomplexarray(munc.get_value(x), covariance=munc.get_covariance(x) + cov)
             for x,cov in zip(Mi, cov_ereff_Gamma(ereff_Gamma, munc.get_value(lengths), X, k, f))]
    
    M     = np.array([x.flatten('F') for x in Mi]).T     # line measurements
    MinvT = np.array([inv(x).flatten('F') for x in Mi])  # inverse line measurements
    
    ## compute W from Takagi factorization
    G, lambd = compute_G_with_takagi(MinvT.dot(M[[0,2,1,3]]), metas=True)
    W = conj((G@np.array([[0,1j],[-1j,0]])).dot(G.T))
    
    # estimated gamma to be used to resolve the sign of W
    gamma_est = 2*np.pi*f/c0*np.sqrt(-(ereff_est-1j*np.finfo(float).eps))  # the eps is to ensure positive square-root
    y_est = np.exp(gamma_est*get_value(lengths))
    z_est = 1/y_est
    if np.sign((y_est.dot(get_value(W)).dot(z_est)).real)-1:
        W = -W # resolve the sign ambiguity
    
    ## Solving the weighted eigenvalue problem
    F = M.dot(W).dot(MinvT[:,[0,2,1,3]]) # weighted measurements
    eigvec, eigval = eig(F) # metas order
    inx = np.argsort(get_value(eigval.real)) # get indices of sorted eigenvalues
    lambd = (eigval[inx[-1]] - eigval[inx[0]])/2   # lambda for debugging; over ride lambd from G (both should be the same)
    
    x1_ = eigvec[:,inx[0]]
    x4  = eigvec[:,inx[-1]]
    x1_ = x1_/x1_[0]
    x4  = x4/x4[-1]
    x2_ = np.array([x4[2], ucomplex(1), x4[2]*x1_[2], x1_[2]])
    x3_ = np.array([x4[1], x4[1]*x1_[1], ucomplex(1), x1_[1]])
    
    X_ = np.array([x1_, x2_, x3_, x4]).T  # normalized calibration matrix
    
    ## Compute progataion constant
    gamma = compute_gamma(X_, M, lengths, gamma_est, metas=True)
    gamma = gamma.real*np.sign(get_value(gamma.real)) + 1j*gamma.imag # make sure to get the correct sign for losses
    ereff = -(c0/2/np.pi/f*gamma)**2
    
    ## De-normalization
    # solve for a11b11 and k from Thru measurement
    ka11b11,_,_,k = solve(X_, M[:,0])
    #ka11b11,_,_,k = inv(X_).dot(M[:,0])
    a11b11  = ka11b11/k
    # solve for a11/b11, a11 and b11
    Ga = Sreflect[0,0]
    Gb = Sreflect[1,1]
    a11_b11 = (reflect_est_b/reflect_est_a)*(Ga - x2_[0])/(1 - Ga*x3_[3])*(1 + Gb*x2_[3])/(Gb + x3_[0])
    a11 = sqrt(a11_b11*a11b11)
    # choose correct answer for a11 and b11
    G0 = get_value( (Ga - x2_[0])/(1 - Ga*x3_[3])/a11 )
    if abs(G0 - get_value(reflect_est_a)) > abs(-G0 - get_value(reflect_est_a)):
        a11 = -a11
    b11 = a11b11/a11
    reflect_est_a = (Ga - x2_[0])/(1 - Ga*x3_[3])/a11  # new value
    reflect_est_b = (Gb + x3_[0])/(1 + Gb*x2_[3])/b11  # new value
    
    # build the calibration matrix (de-normalize)
    X  = dot(X_,np.diag([a11b11, b11, a11, ucomplex(1)]) )
        
    return X, k, get_value(ereff), gamma, get_value(reflect_est_a), get_value(reflect_est_b), lambd

def convert2cov(x, f_length, cov_length=2):
    '''
    make input into covariance matrix
    f_length is the number of frequeny points
    cov_length is the final diagonal length of the cov matrix
    
    Three cases are considerd:
        1. the input is a scalar variance --> convert to diagonal with same variance --> repeat along frequency axes
        2. the input is a vector variance --> only diagonalize it --> repeat along frequency axes
        3. the input is 2D matrix --> do nothing --> repeat along frequency axes
        4. the input is 3D array --> do nothing --> do nothing (assuming the user knows what he/she is doing!)
    '''
    f_length   = int(f_length)
    cov_length = int(cov_length)
    x = np.atleast_1d(x)
    
    if len(x.shape) > 1:
        if len(x.shape) > 2:
            cov = x  # assume everything is fine
        else:
            cov = np.tile(x, (f_length,1,1))
    else:
        if x.shape[0] > 1:
            cov = np.tile(np.diag(x), (f_length,1,1))
        else:
            cov = np.tile(np.eye(cov_length)*x[0], (f_length,1,1))
    
    return cov

class umTRL:
    """
    
    Multiline TRL calibration with uncertainty capabilities, hence "u"mTRL.
    
    """
    
    def __init__(self, lines, line_lengths, reflect, 
                 reflect_est=-1, reflect_offset=0, ereff_est=1+0j, switch_term=None,
                 uSlines=None, ulengths=None, uSreflect=None, ureflect=None, uereff_Gamma=None, uswitch_term=None):
        """
        umTRL initializer.
        """

        self.f  = lines[0].frequency.f
        self.Slines = np.array([x.s for x in lines])
        self.lengths = np.array(line_lengths)
        self.Sreflect = reflect.s
        self.reflect_est = reflect_est
        self.reflect_offset = reflect_offset
        self.ereff_est = ereff_est
        
        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.array([self.f*0 for x in range(2)])
        
        # uncertainties
        self.uSlines      = np.array(uSlines) if uSlines is not None else np.zeros(len(self.lengths))
        if len(self.uSlines.shape) < 3:
            # handle cases when user only give one variance/covaraince for all measurements
            if len(self.uSlines.shape) < 2:
                if len(self.uSlines.shape) < 1:
                    self.uSlines  = np.array([self.uSlines.squeeze() for l in self.lengths])
            else:
                self.uSlines  = np.array([self.uSlines.squeeze() for l in self.lengths])
            
        self.ulengths     = ulengths if ulengths is not None else 0
        self.uSreflect    = uSreflect if uSreflect is not None else 0
        self.ureflect     = ureflect if ureflect is not None else 0
        self.uereff_Gamma = uereff_Gamma if uereff_Gamma is not None else 0
        self.usw          = uswitch_term if uswitch_term is not None else 0
        
    
    def run_mTRL(self):
        # This runs the standard mTRL without uncertainties (very fast).
        gammas  = []
        lambds  = []
        Xs      = []
        ks      = []
        ereff0  = self.ereff_est
        gamma0  = 2*np.pi*self.f[0]/c0*np.sqrt(-ereff0)
        gamma0  = gamma0*np.sign(gamma0.real) # use positive square root
        reflect_est0 = self.reflect_est*np.exp(-2*gamma0*self.reflect_offset)
        reflect_est_a = reflect_est0
        reflect_est_b = reflect_est0
        
        lengths = self.lengths

        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Sreflect = self.Sreflect[inx,:,:]
            sw = self.switch_term[:,inx]
            
            X,k,ereff0,gamma,reflect_est_a,reflect_est_b,lambd = mTRL_at_one_freq(Slines, lengths, Sreflect, 
                                                    ereff_est=ereff0, reflect_est_a=reflect_est_a, 
                                                    f=f, sw=sw, reflect_est_b=reflect_est_b)
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds.append(lambd)
            print(f'Frequency: {f*1e-9:0.2f} GHz')

        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd = np.array(lambds)
        self.error_coef()
        
    def run_umTRL(self):
        # This runs the mTRL with uncertainties 
        # (quite slow because of METAS UncLib, but faster than a MC analysis ;) ).
        
        self.run_mTRL() # initial run to get nominal cal coefficients
        
        gammas  = []
        lambds  = []
        Xs      = []
        ks      = []
        ereff0  = self.ereff[0]
        gamma0  = 2*np.pi*self.f[0]/c0*np.sqrt(-ereff0)
        reflect_est0 = self.reflect_est*np.exp(-2*gamma0*self.reflect_offset)
        reflect_est_a = reflect_est0
        reflect_est_b = reflect_est0
        # line lengths
        lengths = self.lengths
        
        uSlines_full      = np.array([ convert2cov(x, len(self.f), 8) for x in self.uSlines ])
        ulengths_full     = convert2cov(self.ulengths, len(self.f), len(lengths))
        uSreflect_full    = convert2cov(self.uSreflect, len(self.f), 8)
        ureflect_full     = convert2cov(self.ureflect, len(self.f), 2)
        uereff_Gamma_full = convert2cov(self.uereff_Gamma, len(self.f), 4)
        usw_full          = convert2cov(self.usw, len(self.f), 4)
        
        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Sreflect = self.Sreflect[inx,:,:]
            sw = self.switch_term[:,inx]
            
            # uncertainties
            uSlines   = uSlines_full[:,inx,:,:]
            ulengths  = ulengths_full[inx,:,:]
            uSreflect = uSreflect_full[inx,:,:]
            ureflect  = ureflect_full[inx,:,:]
            uereff_Gamma = uereff_Gamma_full[inx,:,:]
            usw   = usw_full[inx,:,:]
            
            X,k,ereff0,gamma, reflect_est_a, reflect_est_b, lambd = \
                umTRL_at_one_freq(Slines, lengths, Sreflect, 
                                  ereff_est=ereff0, reflect_est_a=reflect_est_a, 
                                  reflect_est_b=reflect_est_b, f=f,
                                  X=self.X[inx], k=self.k[inx], sw=sw,
                                  uSlines=uSlines, ulengths=ulengths, 
                                  uSreflect=uSreflect, ureflect=ureflect,
                                  uereff_Gamma=uereff_Gamma, usw=usw
                                  )
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds.append(lambd)
            print(f'Frequency: {f*1e-9:0.2f} GHz')

        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd = np.array(lambds)
        self.error_coef()
    
    def error_coef(self):
        '''
        Return the conventional 12 error terms from the error-box model. The conversion equations are adapted from [4]. Also [5] is a good reference for the equations.
        Originally, I only included the 3 error terms from each port. However, thanks to @Zwelckovich feedback, I decided to update this function to return all 12 error terms. 
        I also included the switch terms for sake of completeness, as well as the consistency test between 8-terms and 12-terms models, as discussed in [4].  

        [4] R. B. Marks, "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," 50th ARFTG Conference Digest, 1997, pp. 115-126.
        [5] Dunsmore, J.P.. Handbook of Microwave Component Measurements: with Advanced VNA Techniques.. Wiley, 2020.

        Below are the error term abbreviations in full. In Marks's paper [4] he just used the abbreviations as is, which can be 
        difficult to understand if you are not familiar with VNA calibration terminology. For those interested in VNAs in general, 
        I recommend the book by Dunsmore [5], where he lists the terms in full.
        
        Left port error terms (forward direction):
        EDF: forward directivity
        ESF: forward source match
        ERF: forward reflection tracking
        ELF: forward load match
        ETF: forward transmission tracking
        EXF: forward crosstalk
        
        Right port error terms (reverse direction):
        EDR: reverse directivity
        ESR: reverse source match
        ERR: reverse reflection tracking
        ELR: reverse load match
        ETR: reverse transmission tracking
        EXR: reverse crosstalk
        
        Switch terms:
        GF: forward switch term
        GR: reverse switch term

        NOTE: the K in my notation is equivalent to Marks' notation [4] by this relationship: K = (beta/alpha)*(1/ERR).
        '''

        self.coefs = {}
        # forward 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDF =  self.X[:,2,3]
        ESF = -self.X[:,3,2]
        ERF =  self.X[:,2,2] - self.X[:,2,3]*self.X[:,3,2]
        
        # reverse 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDR = -self.X[:,1,3]
        ESR =  self.X[:,3,1]
        ERR =  self.X[:,1,1] - self.X[:,3,1]*self.X[:,1,3]
        
        # switch terms
        GF = self.switch_term[0]
        GR = self.switch_term[1]

        # remaining forward terms
        ELF = ESR + ERR*GF/(1-EDR*GF)  # eq. (36) in [4].
        ETF = 1/self.k/(1-EDR*GF)      # eq. (38) in [4], after substituting eq. (36) in eq. (38) and simplifying.
        EXF = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # remaining reverse terms
        ELR = ESF + ERF*GR/(1-EDF*GR)    # eq. (37) in [4].
        ETR = self.k*ERR*ERF/(1-EDF*GR)  # eq. (39) in [4], after substituting eq. (37) in eq. (39) and simplifying.
        EXR = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # forward direction
        self.coefs['EDF'] = EDF
        self.coefs['ESF'] = ESF
        self.coefs['ERF'] = ERF
        self.coefs['ELF'] = ELF
        self.coefs['ETF'] = ETF
        self.coefs['EXF'] = EXF
        self.coefs['GF']  = GF

        # reverse direction
        self.coefs['EDR'] = EDR
        self.coefs['ESR'] = ESR
        self.coefs['ERR'] = ERR
        self.coefs['ELR'] = ELR
        self.coefs['ETR'] = ETR
        self.coefs['EXR'] = EXR
        self.coefs['GR']  = GR

        # consistency check between 8-terms and 12-terms model. Based on eq. (35) in [4].
        # This should equal zero, otherwise there is inconsistency between the models (can arise from switch term measurements).
        self.coefs['check'] = abs( ETF*ETR - (ERR + EDR*(ELF-ESR))*(ERF + EDF*(ELR-ESF)) )
        
        return self.coefs
    
    def apply_cal(self, NW, left=True):
        # apply calibration to a 1-port or 2-port network.
        # NW:   the network to be calibrated (1- or 2-port).
        # left: boolean: define which port to use when 1-port network is given.
        # If left is True, left port is used; otherwise right port is used.
        # The outputs are the corrected NW and its cov matrix (at every freq.)
        
        nports = np.sqrt(len(NW.port_tuples)).astype('int') # number of ports
        # if 1-port, convert to 2-port (later convert back to 1-port)
        if nports < 2:
            NW = rf.two_port_reflect(NW)
        
        metas = True if isinstance(self.k[0], type(munc.ucomplex(0))) else False
        
        # numpy or metas functions
        dot, inv, eig, solve, \
            conj, exp, log, acosh, sqrt, \
                get_value, ucomplex = metas_or_numpy_funcs(metas=metas)
        
        # apply cal
        S_cal = []
        cov   = []
        for x,k,s,sw in zip(self.X, self.k, NW.s, self.switch_term.T):
            s    = correct_switch_term(s, sw[0], sw[1]) if np.any(sw) else s
            xinv = inv(x)
            s11 = ucomplex(complex(s[0,0]))
            s21 = ucomplex(complex(s[1,0]))
            s12 = ucomplex(complex(s[0,1]))
            s22 = ucomplex(complex(s[1,1]))
            M_ = np.array([-s11*s22+s12*s21, -s22, s11, 1])
            T_ = dot(xinv, M_)
            s21_cal = k*s21/T_[-1]
            T_ = T_/T_[-1]
            scal = np.array([[T_[2], (T_[0]-T_[2]*T_[1])/s21_cal],[s21_cal, -T_[1]]])
            if metas:
                S_cal.append(munc.get_value(scal))
                cov.append(munc.get_covariance(scal))
            else:
                S_cal.append(scal)
                cov.append(np.zeros((scal.size*2, scal.size*2)))
            
        S_cal = np.array(S_cal)
        cov   = np.array(cov)
                
        # revert to 1-port device if the input was a 1-port device
        if nports < 2:
            if left: # left port
                S_cal = S_cal[:,0,0]
            else:  # right port
                S_cal = S_cal[:,1,1]
        
        return rf.Network(frequency=NW.frequency, s=S_cal.squeeze()), cov
    
    def shift_plane(self, d=0):
        # shift calibration plane by distance d
        # negative: shift toward port
        # positive: shift away from port
        # e.g., if your Thru has a length of L, 
        # then d=-L/2 to shift the plane backward 
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, self.gamma):
            z = np.exp(-g*d)
            KX_new = k*x.dot(np.diag([z**2, 1, 1, 1/z**2]))
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])
            
        self.X = np.array(X_new)
        self.K = np.array(K_new)
    
    def renorm_impedance(self, Z_new, Z0=50):
        # re-normalize reference calibration impedance
        # by default, the ref impedance is the characteristic 
        # impedance of the line standards.
        # Z_new: new ref. impedance (can be array if frequency dependent)
        # Z0: old ref. impedance (can be array if frequency dependent)
        
        # ensure correct array dimensions (if not, you get an error!)
        N = len(self.k)
        Z_new = Z_new*np.ones(N)
        Z0    = Z0*np.ones(N)
        
        G = (Z_new-Z0)/(Z_new+Z0)
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, G):
            KX_new = k*x.dot( np.kron([[1, -g],[-g, 1]],[[1, g],[g, 1]])/(1-g**2) )
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])

        self.X = np.array(X_new)
        self.K = np.array(K_new)

# EOF