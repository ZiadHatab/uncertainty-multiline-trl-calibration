"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Multiline TRL calibration with linear uncertainty propagation, hence "u"mTRL.

Uncertainty-enabled version of my TUG mTRL algorithm [1-4] (plain implementation:
https://github.com/ZiadHatab/multiline-trl-calibration). mTRL_at_one_freq is numpy;
umTRL_at_one_freq is the same algorithm through METAS UncLib [5] to propagate covariances of
S-parameter noise, line length, reflect asymmetry, line mismatch and switch terms.

[1] Z. Hatab, M. Gadringer and W. Bösch, "Improving The Reliability of The Multiline TRL
    Calibration Algorithm," 98th ARFTG Conf., 2022, doi: 10.1109/ARFTG52954.2022.9844064.
[2] Z. Hatab, M. E. Gadringer and W. Bösch, "Propagation of Linear Uncertainties through
    Multiline TRL Calibration," IEEE TIM, vol. 72, 2023, doi: 10.1109/TIM.2023.3296123.
[3] Z. Hatab, M. E. Gadringer and W. Bösch, "A Thru-Free Multiline Calibration," IEEE TIM,
    vol. 72, 2023, doi: 10.1109/TIM.2023.3308226.
[4] Z. Hatab, M. E. Gadringer and W. Bösch, "The Choice of Line Lengths in Multiline
    Thru-Reflect-Line Calibration," IEEE TIM, vol. 75, 2026, doi: 10.1109/TIM.2026.3704158.
[5] M. Zeier, J. Hoffmann and M. Wollensack, "Metas.UncLib," Metrologia, vol. 49, 2012,
    doi: 10.1088/0026-1394/49/6/809.
"""

# python -m pip install numpy scikit-rf metas_unclib -U
import numpy as np
import skrf as rf
import metas_unclib as munc
munc.use_linprop()

c0 = 299792458.0   # speed of light in vacuum (m/s)

def metas_or_numpy_funcs(metas=False):
    # my way to switch between metas and numpy functions
    if metas:
        dot   = munc.ulinalg.dot
        inv   = munc.ulinalg.inv
        eig   = munc.ulinalg.eig
        conj  = munc.umath.conj
        exp   = munc.umath.exp
        log   = munc.umath.log
        sqrt  = munc.umath.sqrt
        get_value = munc.get_value
        ucomplex  = munc.ucomplex
    else:
        dot   = np.dot
        inv   = np.linalg.inv
        eig   = np.linalg.eig
        conj  = np.conj
        exp   = np.exp
        log   = np.log
        sqrt  = np.sqrt
        get_value = lambda x: x
        ucomplex  = complex
    return dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex

def correct_switch_term(S, GF, GR):
    # remove switch-term effect from measured S-parameters (GF: forward, GR: reverse)
    Sn = S.copy()
    d = 1 - S[0,1]*S[1,0]*GF*GR
    Sn[0,0] = (S[0,0] - S[0,1]*S[1,0]*GF)/d
    Sn[0,1] = (S[0,1] - S[0,0]*S[0,1]*GR)/d
    Sn[1,0] = (S[1,0] - S[1,1]*S[1,0]*GF)/d
    Sn[1,1] = (S[1,1] - S[0,1]*S[1,0]*GR)/d
    return Sn

def s2t(S, pseudo=False):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1] - S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S if pseudo else S/T[1,1]

def sqrt_unwrapped(z):
    # square root of a complex array with its phase unwrapped across frequency
    return np.sqrt(abs(z))*np.exp(0.5j*np.unwrap(np.angle(z)))

def error_matrix(A, B, inv):
    # 16-term error matrix from 2x2 blocks A and P2 inv(B) P2; the P/P2 permutations are
    # done by re-indexing (faster than matrix products in metas).
    E = np.zeros((4,4), dtype=object if np.asarray(A).dtype == object else complex)
    E[:2,:2] = A
    E[2:,2:] = inv(B)[::-1][:,::-1]     # P2 @ inv(B) @ P2
    return E[[0,2,1,3]][:,[0,2,1,3]]    # P.T @ E @ P

def LFTinv(E, S, dot, inv):
    # inverse linear fractional transformation (de-embedding), see Speciale (1981)
    E11, E12, E21, E22 = E[:2,:2], E[:2,2:], E[2:,:2], E[2:,2:]
    return dot(inv(dot(S, E21) - E11), E12 - dot(S, E22))

def compute_G_with_takagi(A, metas=False):
    # Takagi factorization of complex-symmetric A via eig(A A^H) = U diag(s^2) U^H
    # https://www.sciencedirect.com/science/article/pii/S0096300314002239
    dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
    if metas:
        u, s = eig(dot(A, conj(A).T))   # metas eig returns (vectors, values); numpy the reverse
    else:
        s, u = eig(dot(A, conj(A).T))
    sv  = np.sqrt(np.abs(get_value(s)))  # singular values, nominal (lambd is only used nominally)
    inx = np.flip(np.argsort(sv))
    lambd = sv[inx[0]]*sv[inx[1]]
    u = u[:, inx][:, :2]                 # low-rank truncation (Eckart-Young)
    phi = sqrt(conj(np.diag(dot(dot(u.T, conj(A)), u))))
    return dot(u, np.diag(phi)), lambd

def rank1_recover(R, metas=False):
    # dominant left singular vector u1 and its projection u1^H @ R, via eig(R R^H)
    dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
    if metas:
        u, s = eig(dot(R, conj(R).T))
    else:
        s, u = eig(dot(R, conj(R).T))
    u1 = u[:, np.argmax(abs(get_value(s)))]
    return u1, conj(u1).dot(R)

def WLS(x, y, w=1, metas=False):
    # weighted least-squares for a single complex parameter
    dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
    x = x*(1+0j)
    xw = conj(x.dot(w))
    return xw.dot(y)/xw.dot(x)

def Vgl(N):
    # inverse covariance matrix for the propagation-constant fit
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones((N-1, N-1), dtype=complex)

def compute_gamma(z, lengths, gamma_est, metas=False, inx=None):
    # gamma = alpha + 1j*beta from z = exp(-gamma*length) by weighted least-squares.
    # reference line inx minimizes the largest baseline (best phase-unwrap margin).
    dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
    lv = get_value(lengths)
    if inx is None:
        inx = np.argmin([abs(lv - l).max() for l in lv])
    lengths = lengths - lengths[inx]
    z = z/z[inx]
    keep = np.arange(len(lv)) != inx  # exclude the reference line from the fit
    l = -lengths[keep]
    gamma_l = log(z[keep])
    n = np.round((get_value(gamma_l) - gamma_est*get_value(l)).imag/np.pi/2)   # unwrap
    gamma_l = gamma_l - 1j*2*np.pi*n
    gamma = WLS(l, gamma_l, Vgl(int(keep.sum()) + 1), metas)
    return conj(gamma) if get_value(gamma).imag < 0 else gamma   # positive delay (causality)

def compute_lambd(gamma, lengths):
    # squared Frobenius norm of the ideal weighting matrix (used to rank the two gammas)
    z = np.exp(-gamma*lengths)
    W = (np.outer(1/z, z) - np.outer(z, 1/z)).conj()
    return abs(W.conj()*W).sum()/2

def solve_quadratic(v1, v2, inx, x_est, metas=False):
    # recover a calibration column as c1*v1 + c2*v2 by solving the induced quadratic
    dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
    v12, v13 = v1[inx]
    v22, v23 = v2[inx]
    mask = np.ones(v1.shape, bool)
    mask[inx] = False
    v11, v14 = v1[mask]
    v21, v24 = v2[mask]
    k2 = v11*v14*v22**2 + v12**2*v21*v24 - v12*v22*(v11*v24 + v14*v21)
    if abs(get_value(v12)) > abs(get_value(v22)):   # avoid dividing by small numbers
        k1 = -2*v11*v14*v22 - v12**2*v23 + v12*(v11*v24 + v13*v22 + v14*v21)
        k0 = v11*v14 - v12*v13
        c2 = np.array([(-k1 - sqrt(k1**2 - 4*k0*k2))/(2*k2), (-k1 + sqrt(k1**2 - 4*k0*k2))/(2*k2)])
        c1 = (1 - c2*v22)/v12
    else:
        k1 = -2*v12*v21*v24 - v13*v22**2 + v22*(v11*v24 + v12*v23 + v14*v21)
        k0 = v21*v24 - v22*v23
        c1 = np.array([(-k1 - sqrt(k1**2 - 4*k0*k2))/(2*k2), (-k1 + sqrt(k1**2 - 4*k0*k2))/(2*k2)])
        c2 = (1 - c1*v12)/v22
    x = np.array([v1*a + v2*b for a, b in zip(c1, c2)])   # 2 candidate solutions
    return x[np.argmin(abs(get_value(x) - x_est).sum(axis=1))]

def cov_ereff_Gamma(ereff_Gamma, lengths, X, k, f):
    # zero-mean perturbation of the line T-parameters M due to line mismatch (ereff, Gamma).
    # X, k are the nominal error-box coefficients (numpy) from the standard mTRL run.
    dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas=True)
    def Rkron(G):
        d = 1 - G**2
        return np.array([[1/d,     G/d,     -G/d,     -G**2/d],
                         [G/d,     1/d,     -G**2/d,  -G/d],
                         [-G/d,    -G**2/d, 1/d,      G/d],
                         [-G**2/d, -G/d,    G/d,      1/d]])
    cols = []
    for eg, l in zip(ereff_Gamma, lengths):
        g = 2*np.pi*f/c0*sqrt(-eg[0])
        t = np.array([exp(-g*l), ucomplex(0), ucomplex(0), exp(g*l)])
        cols.append(k*X.dot(Rkron(eg[1])).dot(t))
    Mprime = np.array(cols).T
    return Mprime - get_value(Mprime)

def mTRL_at_one_freq(Slines, lengths, Sreflect, gamma_est, reflect_est, reflect_offset,
                     sw=[0,0], compensate_repeated_lines=False, lnorm=1):
    """
    Standard mTRL at a single frequency (no uncertainties, plain numpy).

    Slines         : list of 2x2 line S-parameters (first line is the Thru)
    lengths        : 1D array of line lengths
    Sreflect       : list of 2x2 reflect S-parameters (may be several; nan-filled if none)
    gamma_est      : estimated propagation constant (seeds the sign and phase unwrap)
    reflect_est    : 1D array of reference reflection coefficients (one per reflect)
    reflect_offset : 1D array of reflect offsets relative to the Thru (one per reflect)
    sw             : [forward, reverse] switch terms
    compensate_repeated_lines : True to down-weight repeated lengths in the eigenvalue problem
    lnorm          : 1 for Frobenius norm, 2 for spectral norm (eigenvalue problem scaling)
    """
    reflect_est, reflect_offset = np.atleast_1d(reflect_est), np.atleast_1d(reflect_offset)
    has_reflect = not np.isnan(Sreflect[0][0,0])

    # switch-term correction
    if np.any(sw):
        Slines = [correct_switch_term(x, sw[0], sw[1]) for x in Slines]
        if has_reflect:
            Sreflect = [correct_switch_term(x, sw[0], sw[1]) for x in Sreflect]

    lengths = np.array(lengths) - lengths[0]   # Thru is the reference

    # line T-parameters and their inverses
    Mi    = [s2t(x) for x in Slines]
    M     = np.array([x.flatten('F') for x in Mi]).T
    MinvT = np.array([np.linalg.inv(x).flatten('F') for x in Mi])

    # weighting matrix W from the Takagi factorization (index reorder instead of P@Q)
    G, lambd = compute_G_with_takagi(MinvT@M[[0,2,1,3]])
    W = (G@np.array([[0,1j],[-1j,0]])@G.T).conj()

    # z = exp(-gamma*length) from G, with the sign of W resolved against gamma_est
    eigval, eigvec = np.linalg.eig(G@np.array([[1,-1j],[1j,1]])@G.T)
    z = eigvec[:, np.argmax(abs(eigval))]
    z_est = np.exp(-gamma_est*lengths)
    lambd_est = (1/z_est)@W@z_est
    if abs(lambd_est - lambd) > abs(lambd_est + lambd):
        W, z = -W, 1/z

    # scale W: S1 down-weights repeated lengths, S2 sets the eigenvalue-problem norm [4]
    # (each defaults to identity, so either applies independently).
    _, ui, counts = np.unique(lengths, return_inverse=True, return_counts=True)
    S1 = np.outer(1/counts[ui], 1/counts[ui]) if compensate_repeated_lines else 1
    S2 = abs(W)**(lnorm - 1)
    WS = W*(S1*S2)
    lambd_S = 0.5*abs(WS.conj()*W).sum()   # eigenvalue and its normalization after scaling
    kappa_S = 2*lambd_S/abs(WS).sum()

    # weighted eigenvalue problem -> normalized error terms
    eigval, eigvec = np.linalg.eig(M@WS@MinvT[:,[0,2,1,3]])
    v1, v2, v3, v4 = eigvec[:, np.argsort(eigval.real)].T   # eigenvalues [-lambda,0,0,+lambda]

    x1__est = v1/v1[0]; x1__est[-1] = x1__est[1]*x1__est[2]
    x4_est  = v4/v4[-1]; x4_est[0]  = x4_est[1]*x4_est[2]
    x2__est = np.array([x4_est[2], 1, x4_est[2]*x1__est[2], x1__est[2]])
    x3__est = np.array([x4_est[1], x4_est[1]*x1__est[1], 1, x1__est[1]])
    x1_ = solve_quadratic(v1, v4, [0,3], x1__est)
    x2_ = solve_quadratic(v2, v3, [1,2], x2__est)
    x3_ = solve_quadratic(v2, v3, [2,1], x3__est)
    x4  = solve_quadratic(v1, v4, [3,0], x4_est)

    # build the normalized error boxes (average of range- and null-space answers)
    a12, b21 = (x2_[0] + x4[2])/2, (x3_[0] + x4[1])/2
    a21_a11, b12_b11 = (x1_[1] + x3_[3])/2, (x1_[2] + x2_[3])/2
    A_ = np.array([[1, a12], [a21_a11, 1]])
    B_ = np.array([[1, b12_b11], [b21, 1]])
    X_ = np.kron(B_.T, A_)
    E_ = error_matrix(A_, B_, np.linalg.inv)

    # de-embed the lines and recover s21 = exp(-gamma*length) by rank-1 recovery
    Slines_cal = np.array([LFTinv(E_, s, np.dot, np.linalg.inv) for s in Slines])
    R = np.array([Slines_cal[:,1,0], Slines_cal[:,0,1]])
    _, vh1 = rank1_recover(R)
    s21 = vh1/vh1[0]

    # gamma from the lines (returned to the user) and gamma1 from G (seed for the next point)
    gamma  = compute_gamma(s21, lengths, gamma_est)
    gamma1 = compute_gamma(z,   lengths, gamma_est)
    if abs(compute_lambd(gamma, lengths) - lambd) < abs(compute_lambd(gamma1, lengths) - lambd):
        gamma1 = gamma

    # Thru normalization using S-parameters [3]
    k = 1/Slines_cal[0][1,0]
    a11b11 = Slines_cal[0][0,1]/k

    # solve a11, b11 from the reflect(s); rank-1 recovery combines multiple reflects
    if not has_reflect:
        a11 = b11 = np.sqrt(a11b11)
    else:
        Sreflect_cal = np.array([LFTinv(E_, s, np.dot, np.linalg.inv) for s in Sreflect])
        u1, _ = rank1_recover(np.array([Sreflect_cal[:,0,0], Sreflect_cal[:,1,1]]))
        a11 = np.sqrt(u1[0]/u1[1]*a11b11)
        b11 = a11b11/a11
        G_cal = (Sreflect_cal[:,0,0]/a11 + Sreflect_cal[:,1,1]/b11)/2
        reo = reflect_est*np.exp(-2*gamma1*reflect_offset)
        if abs(G_cal + reo).sum() < abs(G_cal - reo).sum():
            G_cal, a11, b11 = -G_cal, -a11, -b11
        reflect_est = G_cal*np.exp(2*gamma1*reflect_offset)

    X = X_@np.diag([a11b11, b11, a11, 1])
    return X, k, gamma, gamma1, reflect_est, lambd_S, kappa_S


def umTRL_at_one_freq(Slines, lengths, Sreflect, gamma_est, reflect_est, reflect_offset, f, X, k,
                      sw=[0,0], compensate_repeated_lines=False, lnorm=1,
                      uSlines=None, ulengths=None, uSreflect=None, ureflect=None,
                      uereff_Gamma=None, usw=None):
    """
    Same as mTRL_at_one_freq but with linear uncertainty propagation via METAS UncLib.
    X, k are the nominal error-box coefficients (from mTRL_at_one_freq) needed for the
    line-mismatch covariance. The u* arguments are per-frequency covariance matrices.
    Returns an extra par_package with the METAS input variables (for uncertainty budgets).
    """
    dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas=True)
    reflect_est    = np.atleast_1d(reflect_est)
    reflect_offset = np.atleast_1d(reflect_offset)
    n = len(Slines)
    has_reflect = not np.isnan(get_value(Sreflect[0][0,0]))
    ereff_est = -(c0*gamma_est/(2*np.pi*f))**2   # nominal line permittivity for the mismatch model

    # wrap the inputs as METAS variables carrying their covariance
    Slines = np.array([munc.ucomplexarray(s, covariance=c, desc=f'S_line_{j+1}')
                       for j, (s, c) in enumerate(zip(Slines, uSlines))])
    lengths = munc.ufloatarray(lengths, covariance=ulengths, desc='line_lengths')
    ereff_Gamma = np.array([munc.ucomplexarray([ereff_est, 0], covariance=c, desc=f'mismatch_line_{j+1}')
                            for j, c in enumerate(uereff_Gamma)])
    sw_m = munc.ucomplexarray(sw, covariance=usw, desc='switch_terms')
    if has_reflect:
        Sreflect = np.array([munc.ucomplexarray(s, covariance=c, desc=f'S_reflect_{j+1}')
                             for j, (s, c) in enumerate(zip(Sreflect, uSreflect))])
        ra = munc.ucomplex(reflect_est[0], covariance=ureflect)
        rb = munc.ucomplex(reflect_est[0], covariance=ureflect)
        reflect_ratio = munc.ucomplex(1+0j, covariance=munc.get_covariance(rb/ra), desc='reflect_ratio')
    else:
        reflect_ratio = 1
    par_package = (Slines, lengths, Sreflect, reflect_ratio, ereff_Gamma, sw_m)

    # switch-term correction
    if np.any(sw):
        Slines = [correct_switch_term(x, sw_m[0], sw_m[1]) for x in Slines]
        if has_reflect:
            Sreflect = [correct_switch_term(x, sw_m[0], sw_m[1]) for x in Sreflect]

    lengths = lengths - lengths[0]   # Thru is the reference
    lv = get_value(lengths)

    # line T-parameters (perturbed by the line-mismatch covariance) and their inverses
    M = np.array([s2t(x).flatten('F') for x in Slines]).T
    M = M + cov_ereff_Gamma(ereff_Gamma, lv, X, k, f)
    Mi    = [M[:, i].reshape((2,2), order='F') for i in range(n)]
    MinvT = np.array([inv(x).flatten('F') for x in Mi])
    Seff  = [t2s(x) for x in Mi]   # S-parameters carrying the same perturbation

    # weighting matrix W from the Takagi factorization (index reorder instead of P@Q)
    G, lambd = compute_G_with_takagi(dot(MinvT, M[[0,2,1,3]]), metas=True)
    W = conj(dot(dot(G, np.array([[0,1j],[-1j,0]])), G.T))

    # z (= exp(-gamma*length) from G) and the W-sign check only feed the nominal estimate
    # gamma1 below, so use nominal G/W here to avoid pointless auto-diff.
    Gv, Wv = get_value(G), get_value(W)
    eigval, eigvec = np.linalg.eig(Gv@np.array([[1,-1j],[1j,1]])@Gv.T)
    z = eigvec[:, np.argmax(abs(eigval))]
    z_est = np.exp(-gamma_est*lv)
    lambd_est = (1/z_est)@Wv@z_est
    if abs(lambd_est - lambd) > abs(lambd_est + lambd):
        W, z = -W, 1/z
    
    # scale W: S1 down-weights repeated lengths, S2 sets the eigenvalue-problem norm [4]
    # (each defaults to identity, so either applies independently).
    _, ui, counts = np.unique(lv, return_inverse=True, return_counts=True)
    S1 = np.outer(1/counts[ui], 1/counts[ui]) if compensate_repeated_lines else 1
    S2 = abs(W)**(lnorm - 1)
    WS  = W*(S1*S2)
    lambd_S = 0.5*abs(WS.conj()*W).sum()   # eigenvalue and its normalization after scaling
    kappa_S = 2*lambd_S/abs(WS).sum()

    # weighted eigenvalue problem -> normalized error terms
    F = dot(M, dot(WS, MinvT[:, [0,2,1,3]]))
    eigvec, eigval = eig(F)
    v1, v2, v3, v4 = [eigvec[:, i] for i in np.argsort(get_value(eigval).real)]

    x1__est = get_value(v1/v1[0]); x1__est[-1] = x1__est[1]*x1__est[2]
    x4_est  = get_value(v4/v4[-1]); x4_est[0]  = x4_est[1]*x4_est[2]
    x2__est = np.array([x4_est[2], 1, x4_est[2]*x1__est[2], x1__est[2]])
    x3__est = np.array([x4_est[1], x4_est[1]*x1__est[1], 1, x1__est[1]])
    x1_ = solve_quadratic(v1, v4, [0,3], x1__est, metas=True)
    x2_ = solve_quadratic(v2, v3, [1,2], x2__est, metas=True)
    x3_ = solve_quadratic(v2, v3, [2,1], x3__est, metas=True)
    x4  = solve_quadratic(v1, v4, [3,0], x4_est,  metas=True)

    # build the normalized error boxes (average of range- and null-space answers)
    a12, b21 = (x2_[0] + x4[2])/2, (x3_[0] + x4[1])/2
    a21_a11, b12_b11 = (x1_[1] + x3_[3])/2, (x1_[2] + x2_[3])/2
    A_ = np.array([[1, a12], [a21_a11, 1]])
    B_ = np.array([[1, b12_b11], [b21, 1]])
    X_ = np.kron(B_.T, A_)
    E_ = error_matrix(A_, B_, inv)

    # de-embed the lines and recover s21 = exp(-gamma*length) by rank-1 recovery
    Slines_cal = [LFTinv(E_, s, dot, inv) for s in Seff]
    R = np.array([[sc[1,0] for sc in Slines_cal], [sc[0,1] for sc in Slines_cal]])
    _, vh1 = rank1_recover(R, metas=True)
    s21 = vh1/vh1[0]

    # gamma from the de-embedded lines: the only gamma with uncertainty, returned to the user
    gamma  = compute_gamma(s21, lengths, gamma_est, metas=True)
    # gamma1 from G is a nominal estimate, used to resolve the reflect sign and seed the next
    # point, so keep it numpy
    gamma1 = compute_gamma(z, lv, gamma_est)
    if abs(compute_lambd(get_value(gamma), lv) - lambd) < abs(compute_lambd(gamma1, lv) - lambd):
        gamma1 = get_value(gamma)

    # Thru normalization using S-parameters [3].
    k = 1/Slines_cal[0][1,0]
    a11b11 = Slines_cal[0][0,1]/k
    a11b11 = a11b11*exp(-2*gamma*lengths[0])
    k = k/exp(gamma*lengths[0])

    # solve a11, b11 from the reflect(s); rank-1 recovery combines multiple reflects
    if not has_reflect:
        a11 = b11 = sqrt(a11b11)
    else:
        Sreflect_cal = [LFTinv(E_, s, dot, inv) for s in Sreflect]
        u1, _ = rank1_recover(np.array([[sc[0,0] for sc in Sreflect_cal],
                                        [sc[1,1] for sc in Sreflect_cal]]), metas=True)
        a11 = sqrt(u1[0]/u1[1]*reflect_ratio*a11b11)
        b11 = a11b11/a11
        G_cal = np.array([(sc[0,0]/a11 + sc[1,1]/b11)/2 for sc in Sreflect_cal])
        reo = reflect_est*np.exp(-2*gamma1*reflect_offset)
        if abs(get_value(G_cal) + reo).sum() < abs(get_value(G_cal) - reo).sum():
            G_cal, a11, b11 = -G_cal, -a11, -b11
        reflect_est = get_value(G_cal)*np.exp(2*gamma1*reflect_offset)

    X = dot(X_, np.diag([a11b11, b11, a11, ucomplex(1)]))
    return X, k, gamma, gamma1, reflect_est, lambd_S, kappa_S, par_package

def convert2cov(x, num_f, cov_length=2):
    """
    Expand a user-supplied uncertainty into a (num_f, cov_length, cov_length) covariance:
        scalar variance -> scaled identity, repeated over frequency
        1D variances    -> diagonal, repeated over frequency
        2D covariance    -> repeated over frequency
        3D array         -> used as-is (already frequency-dependent)
    """
    num_f, cov_length = int(num_f), int(cov_length)
    x = np.atleast_1d(x)
    if x.ndim > 2:
        return x
    if x.ndim == 2:
        return np.tile(x, (num_f, 1, 1))
    if x.shape[0] > 1:
        return np.tile(np.diag(x), (num_f, 1, 1))
    return np.tile(np.eye(cov_length)*x[0], (num_f, 1, 1))

def per_standard(u, n):
    # normalize an uncertainty spec into a list of n covariance specs (one per standard)
    if u is None:
        return [0.0]*n
    arr = np.asarray(u)
    if arr.ndim >= 3 and arr.shape[0] == n:
        return [arr[i] for i in range(n)]
    return [arr]*n

class umTRL:
    """Multiline TRL calibration with linear uncertainty propagation."""

    def __init__(self, lines, line_lengths, reflect=None,
                 reflect_est=-1, reflect_offset=0, ereff_est=1+0j, switch_term=None,
                 compensate_repeated_lines=False, lnorm=1,
                 uSlines=None, ulengths=None, uSreflect=None, ureflect=None,
                 uereff_Gamma=None, uswitch_term=None):
        self.f       = lines[0].frequency.f
        self.Slines  = np.array([x.s for x in lines])
        self.lengths = np.array(line_lengths)
        self.ereff_est = ereff_est*(1 + 0j)
        self.compensate_repeated_lines = compensate_repeated_lines
        self.lnorm   = lnorm

        # reflect(s): accept a single Network or a list; nan-filled if none given
        if reflect is None:
            self.Sreflect = np.ones((1, len(self.f), 2, 2))*np.nan
        else:
            reflect = reflect if isinstance(reflect, list) else [reflect]
            self.Sreflect = np.array([x.s for x in reflect])
        self.reflect_est    = np.atleast_1d(reflect_est)
        self.reflect_offset = np.atleast_1d(reflect_offset)

        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.zeros((2, len(self.f)), dtype=complex)

        # uncertainties (per-standard specs are expanded to one entry per line/reflect)
        self.uSlines      = per_standard(uSlines, len(self.lengths))
        self.uereff_Gamma = per_standard(uereff_Gamma, len(self.lengths))
        self.uSreflect    = per_standard(uSreflect, self.Sreflect.shape[0])
        self.ulengths     = ulengths if ulengths is not None else 0
        self.ureflect     = ureflect if ureflect is not None else 0
        self.usw          = uswitch_term if uswitch_term is not None else 0

    def run_mTRL(self):
        # standard mTRL without uncertainties (fast, numpy only)
        print('\nmTRL (no uncertainty) running...')
        Xs, ks, gammas, lambds_S, kappas_S = ([] for _ in range(5))
        gamma_est = 2*np.pi*self.f[0]/c0*np.sqrt(-self.ereff_est)
        gamma_est = np.sign(gamma_est.imag)*gamma_est   # seed with positive imag (forward wave)
        for i, f in enumerate(self.f):
            X, k, gamma, gamma_est, _, lambd_S, kappa_S = mTRL_at_one_freq(
                list(self.Slines[:, i]), self.lengths, list(self.Sreflect[:, i]),
                gamma_est, self.reflect_est, self.reflect_offset, sw=self.switch_term[:, i],
                compensate_repeated_lines=self.compensate_repeated_lines, lnorm=self.lnorm)
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds_S.append(lambd_S)
            kappas_S.append(kappa_S)
            if i + 1 < len(self.f):
                gamma_est = gamma_est/f*self.f[i+1]   # scale gamma to the next point (gamma ~ f)
            print(f'Frequency: {f*1e-9:.2f} GHz done!', end='\r', flush=True)
        self._store(Xs, ks, gammas, lambds_S, kappas_S)

    def run_umTRL(self):
        # mTRL with linear uncertainty propagation (uses METAS UncLib; slower)
        self.run_mTRL()   # nominal coefficients, needed for the line-mismatch covariance
        X_nom, k_nom = self.X, self.k
        print('\nmTRL (with uncertainty) running...')

        # per-frequency covariance matrices
        nf = len(self.f)
        uSlines      = [convert2cov(u, nf, 8) for u in self.uSlines]
        uereff_Gamma = [convert2cov(u, nf, 4) for u in self.uereff_Gamma]
        uSreflect    = [convert2cov(u, nf, 8) for u in self.uSreflect]
        ulengths     = convert2cov(self.ulengths, nf, len(self.lengths))
        ureflect     = convert2cov(self.ureflect, nf, 2)
        usw          = convert2cov(self.usw, nf, 4)

        Xs, ks, gammas, lambds_S, kappas_S = ([] for _ in range(5))
        Slines_m, lengths_m, Sreflect_m, reflect_ratio_m, ereff_Gamma_m, sw_m = ([] for _ in range(6))
        gamma_est = 2*np.pi*self.f[0]/c0*np.sqrt(-self.ereff_est)
        gamma_est = np.sign(gamma_est.imag)*gamma_est   # seed with positive imag (forward wave)
        for i, f in enumerate(self.f):
            X, k, gamma, gamma_est, _, lambd_S, kappa_S, par = umTRL_at_one_freq(
                list(self.Slines[:, i]), self.lengths, list(self.Sreflect[:, i]),
                gamma_est, self.reflect_est, self.reflect_offset, f, X_nom[i], k_nom[i],
                sw=self.switch_term[:, i],
                compensate_repeated_lines=self.compensate_repeated_lines, lnorm=self.lnorm,
                uSlines=[u[i] for u in uSlines], ulengths=ulengths[i],
                uSreflect=[u[i] for u in uSreflect], ureflect=ureflect[i],
                uereff_Gamma=[u[i] for u in uereff_Gamma], usw=usw[i])
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds_S.append(lambd_S)
            kappas_S.append(kappa_S)
            Slines_m.append(par[0])
            lengths_m.append(par[1])
            Sreflect_m.append(par[2])
            reflect_ratio_m.append(par[3])
            ereff_Gamma_m.append(par[4])
            sw_m.append(par[5])
            if i + 1 < len(self.f):
                gamma_est = gamma_est/f*self.f[i+1]   # scale gamma to the next point (gamma ~ f)
            print(f'Frequency: {f*1e-9:.2f} GHz done!', end='\r', flush=True)

        self.Slines_metas        = np.array(Slines_m)
        self.lengths_metas       = np.array(lengths_m)
        self.Sreflect_metas      = np.array(Sreflect_m)
        self.reflect_ratio_metas = np.array(reflect_ratio_m)
        self.ereff_Gamma_metas   = np.array(ereff_Gamma_m)
        self.sw_metas            = np.array(sw_m)
        self._store(Xs, ks, gammas, lambds_S, kappas_S)

    def _store(self, Xs, ks, gammas, lambds_S, kappas_S):
        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd_S = np.array(lambds_S)   # same, after the S1/S2 scaling of W
        self.kappa_S = np.array(kappas_S)
        self.error_coef()

    def error_coef(self):
        '''
        Return the conventional 12 error terms of the error-box model, plus the switch terms
        and the 8-vs-12-term consistency check. Notation follows Marks ("Formulations of the
        Basic VNA Error Model including Switch-Terms", 50th ARFTG, 1997); see also Dunsmore
        ("Handbook of Microwave Component Measurements", Wiley, 2020). Here k = (beta/alpha)/ERR.
        '''
        c = {}
        c['EDF'] =  self.X[:,2,3]
        c['ESF'] = -self.X[:,3,2]
        c['ERF'] =  self.X[:,2,2] - self.X[:,2,3]*self.X[:,3,2]
        c['EDR'] = -self.X[:,1,3]
        c['ESR'] =  self.X[:,3,1]
        c['ERR'] =  self.X[:,1,1] - self.X[:,3,1]*self.X[:,1,3]
        c['GF']  = self.switch_term[0]
        c['GR']  = self.switch_term[1]
        c['ELF'] = c['ESR'] + c['ERR']*c['GF']/(1 - c['EDR']*c['GF'])
        c['ETF'] = 1/self.k/(1 - c['EDR']*c['GF'])
        c['EXF'] = 0*c['ESR']
        c['ELR'] = c['ESF'] + c['ERF']*c['GR']/(1 - c['EDF']*c['GR'])
        c['ETR'] = self.k*c['ERR']*c['ERF']/(1 - c['EDF']*c['GR'])
        c['EXR'] = 0*c['ESR']
        # 8-vs-12-term consistency (eq. (35) in Marks); should be ~0
        c['check'] = abs(c['ETF']*c['ETR'] - (c['ERR'] + c['EDR']*(c['ELF']-c['ESR']))
                                             *(c['ERF'] + c['EDF']*(c['ELR']-c['ESF'])))
        self.coefs = c
        return c

    def apply_cal(self, NW, cov=None, left=True):
        # apply the calibration to a 1- or 2-port network.
        # cov: optional per-frequency covariance of NW.s (enables uncertainty on the DUT).
        # left: which port to keep when NW is 1-port. Returns (skrf Network, S-parameters).
        nports = int(np.sqrt(len(NW.port_tuples)))
        if nports < 2:
            NW = rf.two_port_reflect(NW)

        metas = isinstance(self.k[0], type(munc.ucomplex(0))) or (cov is not None)
        dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
        covs = [None]*len(self.f) if cov is None else cov

        S_cal = []
        for X, k, s, sw, c in zip(self.X, self.k, NW.s, self.switch_term.T, covs):
            if c is not None:
                s = munc.ucomplexarray(s, covariance=c)
            s = correct_switch_term(s, sw[0], sw[1]) if np.any(sw) else s
            A = np.array([[X[2,2], X[2,3]], [X[3,2], 1]])
            B = np.array([[X[1,1], X[3,1]], [X[1,3], 1]])
            S_cal.append(LFTinv(error_matrix(k*A, B, inv), s, dot, inv))

        S_cal = np.array(S_cal)
        if nports < 2:
            S_cal = S_cal[:, 0, 0] if left else S_cal[:, 1, 1]
        return rf.Network(frequency=NW.frequency, s=get_value(S_cal).squeeze()), S_cal.squeeze()

    def reciprocal_ntwk(self):
        # split the calibration into the left and right error-boxes as skrf Networks,
        # assuming they are reciprocal (S21 = S12). Nominal values only (skrf holds no metas).
        val = munc.get_value if isinstance(self.k[0], type(munc.ucomplex(0))) else (lambda x: x)
        freq = rf.Frequency.from_f(self.f, unit='hz')
        freq.unit = 'ghz'

        def box(ED, ES, ER):
            s11, s22 = val(self.coefs[ED]), val(self.coefs[ES])
            s21 = sqrt_unwrapped(val(self.coefs[ER]))   # S21 = S12 by reciprocity
            return np.array([[[a, c], [c, b]] for a, b, c in zip(s11, s22, s21)])

        left  = rf.Network(s=box('EDF', 'ESF', 'ERF'), frequency=freq, name='Left error-box')
        right = rf.Network(s=box('EDR', 'ESR', 'ERR'), frequency=freq, name='Right error-box')
        right.flip()   # so it de-embeds from port-2's perspective
        return left, right

    def shift_plane(self, da=0, db=None):
        # shift the calibration plane by da from port-1 and db from port-2 (db defaults to da).
        # negative shifts toward the port, positive away from it.
        db = da if db is None else db
        metas = isinstance(self.k[0], type(munc.ucomplex(0)))
        dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
        Xs, ks = [], []
        for X, k, g in zip(self.X, self.k, self.gamma):
            KX = k*dot(X, np.diag([exp(-g*(db+da)), exp(-g*(db-da)), exp(g*(db-da)), exp(g*(db+da))]))
            Xs.append(KX/KX[-1,-1]); ks.append(KX[-1,-1])
        self.X, self.k = np.array(Xs), np.array(ks)
        self.error_coef()

    def renorm_impedance(self, Z_new, Z0=50):
        # re-normalize the reference impedance (default: line characteristic impedance).
        # Z_new, Z0 may be scalars or frequency-dependent arrays.
        metas = isinstance(self.k[0], type(munc.ucomplex(0)))
        dot, inv, eig, conj, exp, log, sqrt, get_value, ucomplex = metas_or_numpy_funcs(metas)
        Z_new, Z0 = Z_new*np.ones(len(self.k)), Z0*np.ones(len(self.k))
        G = (Z_new - Z0)/(Z_new + Z0)
        Xs, ks = [], []
        for X, k, g in zip(self.X, self.k, G):
            KX = k*dot(X, np.kron([[1, -g], [-g, 1]], [[1, g], [g, 1]])/(1 - g**2))
            Xs.append(KX/KX[-1,-1]); ks.append(KX[-1,-1])
        self.X, self.k = np.array(Xs), np.array(ks)
        self.error_coef()

# EOF
