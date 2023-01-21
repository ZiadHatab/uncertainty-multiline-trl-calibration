"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Example of comparing mTRL linear uncertainly propagation and Monte Carlo using CPW model.
The simulation is based on the error-boxes from measurements, in which the CPW models are embedded in.
The uncertainty due to the VNA was determined from the multi-sweep as sample covariance.
The uncertainty due to the standards are estimated using CPW model.
"""

import os
import copy
import zipfile

# pip install numpy matplotlib scikit-rf metas_unclib -U
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import metas_unclib as munc
munc.use_linprop()

# umTRL.py and cpw.py should be in same folder 
from umTRL import umTRL
from cpw import CPW

def read_waves_to_S_from_zip(zipfile_full_dir, file_name_contain):
    # read wave parameter files and convert to S-parameters (from a zip file)
    with zipfile.ZipFile(zipfile_full_dir, mode="r") as archive:
        netwks = rf.read_zipped_touchstones(archive)
        A = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_A' in key])
        B = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_B' in key])    
    freq = A[0].frequency
    S = rf.NetworkSet( [rf.Network(s=b.s@np.linalg.inv(a.s), frequency=freq) for a,b in zip(A,B)] )
    return S.mean_s, S.cov(), np.array([s.s for s in S])

class PlotSettings:
    # to make plots look better for publication
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    def __init__(self, font_size=10, latex=False): 
        self.font_size = font_size 
        self.latex = latex
    def __enter__(self):
        plt.style.use('seaborn-v0_8-paper')
        # make svg output text and not curves
        plt.rcParams['svg.fonttype'] = 'none'
        # fontsize of the axes title
        plt.rc('axes', titlesize=self.font_size*1.2)
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=self.font_size)
        # fontsize of the tick labels
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('ytick', labelsize=self.font_size)
        # legend fontsize
        plt.rc('legend', fontsize=self.font_size*1)
        # fontsize of the figure title
        plt.rc('figure', titlesize=self.font_size)
        # controls default text sizes
        plt.rc('text', usetex=self.latex)
        #plt.rc('font', size=self.font_size, family='serif', serif='Times New Roman')
        plt.rc('lines', linewidth=1.5)
    def __exit__(self, exception_type, exception_value, traceback):
        plt.style.use('default')

def s2t(S, pseudo=False):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return [T,S[1,0]] if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return [S,T[1,1]] if pseudo else S/T[1,1]

def Qnm(Zn, Zm):
    # Impedance transformer in T-parameters from on Eqs. (86) and (87) in
    # R. Marks and D. Williams, "A general waveguide circuit theory," 
    # Journal of Research (NIST JRES), National Institute of Standards and Technology,
    # Gaithersburg, MD, no. 97, 1992.
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914227/
    Gnm = (Zm-Zn)/(Zm+Zn)
    return np.sqrt(Zn.real/Zm.real*(Zm/Zn).conjugate())/np.sqrt(1-Gnm**2)*np.array([[1, Gnm],[Gnm, 1]])
    
def TL(l, cpw, Z01=None, Z02=None):
    # create skrf network from a general transmission line model from an cpw object (file: cpw.py)
    N = len(cpw.Z0)  # number of frequency points
    Z01 = cpw.Z0 if Z01 is None else np.atleast_1d(Z01)*np.ones(N)
    Z02 = Z01 if Z02 is None else np.atleast_1d(Z02)*np.ones(N)
    S = []
    for g,zc,z01,z02 in zip(cpw.gamma, cpw.Z0, Z01, Z02):
        T = Qnm(z01,zc)@np.diag([np.exp(-l*g), np.exp(l*g)])@Qnm(zc,z02)
        S.append(t2s(T))
    freq = rf.Frequency.from_f(cpw.f, unit='Hz')
    freq.unit = 'GHz'
    return rf.Network(s=np.array(S), frequency=freq, name=f'l={l*1e3:.2f}mm')

def offset_open(l,cpw, l2=None):
    # create a 2-port offset open network from cpw object (file: cpw.py)
    if l2 is None:
        l2 = l
    freq = rf.Frequency.from_f(cpw.f, unit='Hz')
    freq.unit = 'GHz'
    single_port_1 = rf.Network(s=np.array([np.exp(-2*l*g) for g in cpw.gamma]), frequency=freq, name='open')
    single_port_2 = rf.Network(s=np.array([np.exp(-2*l2*g) for g in cpw.gamma]), frequency=freq, name='open')
    return rf.two_port_reflect(single_port_1,single_port_2)  # make it 2-port (S11=S22)

def ideal_sym_DUT(freq):
    # Equal reflection and transmission network
    s = 1/np.sqrt(2)
    S = np.array([[s, s], [s, s]])
    return rf.Network(s=np.tile(S, (len(freq.f), 1,1)), frequency=freq)

def embbed_error(k,X,NW):
    # embed the error box to an skrf network
    eps = np.finfo(float).eps
    new_NW = NW.copy()
    S = NW.s
    out = [s2t(s,pseudo=True) for s in S]
    T = [x[0] for x in out]
    C = [x[1] for x in out]
    S_scale = np.array([t2s( kk*XX.dot(t.flatten('F')).reshape((2,2), order='F') ) for t,kk,XX in zip(T,k,X)])
    S_new = np.array([ s*np.array([[1,1/(c+eps)],[c+eps,1]]) for s,c in zip(S_scale,C)])
    new_NW.s = S_new
    return new_NW

def add_white_noise(NW, covs):
    # add white noise to a network's S-parameters
    NW_new = NW.copy()
    for inx,(s,cov) in enumerate(zip(NW_new.s,covs)):
        h = np.kron(s.flatten('F').real,[1,0]) + np.kron(s.flatten('F').imag,[0,1])
        noise = np.random.multivariate_normal(np.zeros(h.size), cov)
        E = np.kron(np.eye(len(s)*2), [1,1j])
        NW_new.s[inx] = s + E.dot(noise).reshape((2,2),order='F')
    return NW_new

def get_cov_component(metas_val, para):
    # To get the uncertainty due to each parameter while accounting for their correlation 
    cov = []
    for inx in range(len(metas_val)):
        J = munc.get_jacobi2(metas_val[inx], para[inx])
        U = munc.get_covariance(para[inx])
        cov.append(J@U@J.T)
    return np.array(cov).squeeze()

if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    
    path = os.path.dirname(os.path.realpath(__file__)) + '\\FF_ISS_measurements\\'
    file_name = 'ff_ISS'
    print('Loading files... please wait!!!')
    L1, L1_cov, L1S = read_waves_to_S_from_zip(path + f'{file_name}_thru.zip', f'{file_name}_thru')
    L2, L2_cov, L2S = read_waves_to_S_from_zip(path + f'{file_name}_line01.zip', f'{file_name}_line01')
    L3, L3_cov, L3S = read_waves_to_S_from_zip(path + f'{file_name}_line02.zip', f'{file_name}_line02')
    L4, L4_cov, L4S = read_waves_to_S_from_zip(path + f'{file_name}_line03.zip', f'{file_name}_line03')
    L5, L5_cov, L5S = read_waves_to_S_from_zip(path + f'{file_name}_line04.zip', f'{file_name}_line04')
    L6, L6_cov, L6S = read_waves_to_S_from_zip(path + f'{file_name}_line05.zip', f'{file_name}_line05')
    OPEN, OPEN_cov, OPENS = read_waves_to_S_from_zip(path + f'{file_name}_open.zip', f'{file_name}_open')
    f = L1.frequency.f  # frequency axis
    
    # CPW model parameters 
    w, s, wg, t = 49.1e-6, 25.5e-6, 273.3e-6, 4.9e-6
    Dk = 9.9
    Df = 0.0
    sig = 4.11e7  # conductivity of Gold
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sig)
    cpw.update_jac() # compute the Jacobian of the cpw with respect to its inputs
    
    # mTRL definition
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6, 3500e-6, 5250e-6]
    reflect = OPEN
    reflect_est = 1
    reflect_offset = -0.1e-3
    ereff_est = 5.45-0.0001j

    ## compare the CPW model with measurements
    cal = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=ereff_est )
    cal.run_mTRL() # run normal mTRL without uncertainty
    with PlotSettings(14):
        gamma_mTRL = cal.gamma
        loss_dbmm_mTRL = gamma2dbmm(gamma_mTRL)
        ereff_mTRL = cal.ereff
        fig, axs = plt.subplots(1,2, figsize=(10,3.8))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0]
        ax.plot(f*1e-9, ereff_mTRL.real, lw=2, label='Measurement', 
                marker='^', markevery=15, markersize=10)
        ax.plot(f*1e-9, cpw.ereff.real, lw=2, label='CPW model', 
                marker='v', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([4.5, 6])
        ax.set_yticks(np.arange(4.5, 6.01, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.legend()
        ax = axs[1]
        ax.plot(f*1e-9, gamma2dbmm(gamma_mTRL), lw=2, label='Measurement', 
                marker='^', markevery=15, markersize=10)
        ax.plot(f*1e-9, gamma2dbmm(cpw.gamma), lw=2, label='CPW model', 
                marker='v', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 1.5])
        ax.set_yticks(np.arange(0, 1.51, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.legend()
    
    ## Below is to compare MC with linear uncertainty propagation.
    ## the data for the standards is based on CPW model and the error-box are from the measurements
    ## the uncertainty noise is from measurements, while standards uncertainties are estimated.
    # define a DUT with the error-boxes from the measurements
    DUT = ideal_sym_DUT(L1.frequency)  # ideal lossless symmetric DUT
    DUT_embbed = embbed_error(cal.k, cal.X, DUT) # embedded DUT with the error-boxes
    
    # line lengths
    line_lengths = [x-line_lengths[0] for x in line_lengths]
    reflect_offset = 0
    
    # Noise uncertainties
    uSlines   = np.array([L1_cov, L2_cov, L3_cov, L4_cov, L5_cov, L6_cov]) # measured lines
    uSreflect = OPEN_cov # measured reflect 
    
    # length uncertainties
    l_std = 40e-6  # for the line
    ulengths  = l_std**2  
    l_open_std = 40e-6 # uncertainty in length used for the reflect
    
    # cross-section uncertainties
    w_std   = 2.55e-6
    s_std   = 2.55e-6
    wg_std  = 2.55e-6
    t_std   = 0.49e-6
    Dk_std  = 0.2
    Df_std  = 0
    sig_std = sig*0.1

    # Monte Carlo simulation 
    M = 100 # number of MC runs
    cpw_MC = copy.deepcopy(cpw)
    loss_dbmm_mTRL_model_MC = []
    ereff_real_mTRL_model_MC = []
    DUT_cal_MC = []
    open_model_collect = []
    for m in range(M):
        print(f'MC index {m+1} out of {M}')
        lines_model_MC = []
        for inx,(l,cov) in enumerate(zip(line_lengths, uSlines)):
            cpw_MC.w     = w + np.random.randn()*w_std
            cpw_MC.s     = s + np.random.randn()*s_std
            cpw_MC.wg    = wg + np.random.randn()*wg_std
            cpw_MC.t     = t + np.random.randn()*t_std
            cpw_MC.er    = (Dk + np.random.randn()*Dk_std)*(1-1j*(Df + np.random.randn()*Df_std))
            cpw_MC.sigma = sig + np.random.randn()*sig_std
            cpw_MC.update()
            length = l + np.random.randn()*l_std
            embbed_line = embbed_error(cal.k, cal.X, TL(length,cpw_MC, cpw.Z0))
            lines_model_MC.append( add_white_noise(embbed_line, cov) )
                    
        open_model_MC = offset_open(0 + np.random.randn()*l_open_std, cpw_MC, l2 = 0 + np.random.randn()*l_open_std)    
        open_model_collect.append(open_model_MC)
        reflect_model_MC = add_white_noise(embbed_error(cal.k, cal.X, open_model_MC), uSreflect)
        
        cal_MC = umTRL(lines=lines_model_MC, line_lengths=line_lengths, reflect=reflect_model_MC, 
                   reflect_est=reflect_est, reflect_offset=reflect_offset, ereff_est=ereff_est)
        cal_MC.run_mTRL() # run normal mTRL without uncertainty 
        DUT_cal_MC.append(cal_MC.apply_cal(DUT_embbed)[0])
        loss_dbmm_mTRL_model_MC.append( gamma2dbmm(cal_MC.gamma) )
        ereff_real_mTRL_model_MC.append( cal_MC.ereff.real )
    
    open_model_collect = rf.NetworkSet(open_model_collect)
    DUT_cal_MC = rf.NetworkSet(DUT_cal_MC)
    DUT_cal_MC_cov = DUT_cal_MC.cov()
    
    loss_dbmm_mTRL_model_MC = np.array(loss_dbmm_mTRL_model_MC)
    mu_loss_dbmm_mTRL_model_MC = loss_dbmm_mTRL_model_MC.mean(axis=0)
    std_loss_dbmm_mTRL_model_MC = loss_dbmm_mTRL_model_MC.std(axis=0)
    
    ereff_real_mTRL_model_MC = np.array(ereff_real_mTRL_model_MC)
    mu_ereff_real_mTRL_model_MC = ereff_real_mTRL_model_MC.mean(axis=0)
    std_ereff_real_mTRL_model_MC = ereff_real_mTRL_model_MC.std(axis=0)
    
    # line mismatch uncertainty
    U = np.diag([w_std,s_std,wg_std,t_std,Dk_std,Df_std,sig_std])**2
    uereff_Gamma_i = np.array([ np.vstack((x,y)).dot(U).dot(np.vstack((x,y)).T) for x,y in zip(cpw.jac_ereff,cpw.jac_Gamma)])
    uereff_Gamma   = [uereff_Gamma_i]*len(lines) # repeat for all lines
    
    # open asymmetry
    # the uncertainty is computed analyically as an offset asymmetry between the ports
    diff_open = lambda g,l: -2*g*np.exp(-2*g*l)
    ureflect     = np.array([ np.array([[diff_open(g,reflect_offset).real],[diff_open(g,reflect_offset).imag]]).dot(
        np.array([[diff_open(g,reflect_offset).real],[diff_open(g,reflect_offset).imag]]).T)*l_open_std**2 for g in cpw.gamma])

    # simulated calibration standards
    lines_model   = [ embbed_error(cal.k, cal.X, TL(l,cpw)) for l in line_lengths ]
    reflect_model = embbed_error( cal.k, cal.X, offset_open(0, cpw) )
    
    # umTRL with linear uncertainty evaluation
    cal_lin = umTRL(lines=lines_model, line_lengths=line_lengths, reflect=reflect_model, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est,
               uSlines=uSlines, uSreflect=uSreflect,
               ulengths=ulengths,
               ureflect=ureflect, uereff_Gamma=uereff_Gamma)
    cal_lin.run_umTRL() # run mTRL with linear uncertainty propagation
    _, DUT_cal_metas = cal_lin.apply_cal(DUT_embbed)
    
    # Consider only reflect asymmetry (used later to compare uncertainty contribution)
    # I'm doing this because Metas unclib package gives me an error when 
    # trying to extract the Jacobian with respect to reflect uncertainties
    cal_lin_reflect = umTRL(lines=lines_model, line_lengths=line_lengths, reflect=reflect_model, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, 
               uSlines=uSlines*0, uSreflect=uSreflect*0, ulengths=ulengths*0,
               ureflect=ureflect, uereff_Gamma=np.array(uereff_Gamma)*0)
    cal_lin_reflect.run_umTRL() # run mTRL with linear uncertainty propagation
    _, DUT_cal_metas_reflect = cal_lin_reflect.apply_cal(DUT_embbed)
    
    # comparison between MC and linear propagation
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0,0]
        mu  = munc.get_value(cal_lin.ereff).real
        std = munc.get_stdunc(cal_lin.ereff).real
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation', 
                marker='^', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        mu  = mu_ereff_real_mTRL_model_MC
        std = std_ereff_real_mTRL_model_MC
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL Monte Carlo', 
                marker='v', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, cpw.ereff.real, lw=2, label='True value', color='black')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([4.5, 6])
        ax.set_yticks(np.arange(4.5, 6.01, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[0,1]
        loss_dbmm_mTRL_model_lin = gamma2dbmm(cal_lin.gamma)
        mu  = munc.get_value(loss_dbmm_mTRL_model_lin)
        std = munc.get_stdunc(loss_dbmm_mTRL_model_lin)
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation', 
                marker='^', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        mu  = mu_loss_dbmm_mTRL_model_MC
        std = std_loss_dbmm_mTRL_model_MC
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL Monte Carlo', 
                marker='v', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, gamma2dbmm(cpw.gamma), lw=2, label='True value', color='black')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 1.5])
        ax.set_yticks(np.arange(0, 1.51, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,0]
        mu  = munc.get_value(abs(DUT_cal_metas[:,0,0]))
        std = munc.get_stdunc(abs(DUT_cal_metas[:,0,0]))
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation', 
                marker='^', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        S11_M = np.array([x.s[:,0,0] for x in DUT_cal_MC])
        mu  = abs(S11_M).mean(axis=0)
        std = abs(S11_M).std(axis=0)
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL Monte Carlo', 
                marker='v', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, DUT.s11.s.squeeze().real, lw=2, label='True value', color='black')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S11 (mag)')
        ax.set_ylim([0.5, 0.9])
        ax.set_yticks(np.arange(.5, 0.91, 0.1))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,1]
        mu  = munc.get_value(abs(DUT_cal_metas[:,1,0]))
        std = munc.get_stdunc(abs(DUT_cal_metas[:,1,0]))
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation', 
                marker='^', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        S21_M = np.array([x.s[:,1,0] for x in DUT_cal_MC])
        mu  = abs(S21_M).mean(axis=0)
        std = abs(S21_M).std(axis=0)
        p = ax.plot(f*1e-9, mu, lw=2, label='mTRL Monte Carlo', 
                marker='v', markevery=15, markersize=10)
        ax.plot(f*1e-9, mu-std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, mu+std*k, linestyle=(0, (5, 5)), lw=2, color=p[0].get_color())
        ax.plot(f*1e-9, DUT.s21.s.squeeze().real, lw=2, label='True value', color='black')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S21 (mag)')
        ax.set_ylim([0.5, 0.9])
        ax.set_yticks(np.arange(.5, 0.91, 0.1))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.97), 
                   loc='lower center', ncol=3, borderaxespad=0,
                   title = r"CPW parameters and calibrated DUT with 95% uncertainty bounds ($2\times\sigma$)"
                   )
    
    ## uncertainties due to uncertainty type 
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0,0]
        val_metas = munc.umath.real(cal_lin.ereff)
        val_metas_reflect_only = munc.umath.real(cal_lin_reflect.ereff)  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)                    
        std = np.sqrt(get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                      + get_cov_component(val_metas, cal_lin.Slines_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise', marker='^', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length offset', marker='v', markevery=15, markersize=10)
        std = munc.get_stdunc(val_metas_reflect_only)
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect asymmetry', marker='>', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.ereff_Gamma_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Line mismatch', marker='<', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([0, 0.3])
        ax.set_yticks(np.arange(0, 0.41, 0.1))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[0,1]
        val_metas = gamma2dbmm(cal_lin.gamma)
        val_metas_reflect_only = gamma2dbmm(cal_lin_reflect.gamma)  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)                    
        std = np.sqrt(get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                      + get_cov_component(val_metas, cal_lin.Slines_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise', marker='^', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length offset', marker='v', markevery=15, markersize=10)
        std = munc.get_stdunc(val_metas_reflect_only)
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect asymmetry', marker='>', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.ereff_Gamma_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Line mismatch', marker='<', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 0.2])
        ax.set_yticks(np.arange(0, 0.21, 0.05))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,0]
        val_metas = abs(DUT_cal_metas[:,0,0]).squeeze()
        val_metas_reflect_only = abs(DUT_cal_metas_reflect[:,0,0]).squeeze()  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)                    
        std = np.sqrt(get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                      + get_cov_component(val_metas, cal_lin.Slines_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise', marker='^', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length offset', marker='v', markevery=15, markersize=10)
        std = munc.get_stdunc(val_metas_reflect_only)
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect asymmetry', marker='>', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.ereff_Gamma_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Line mismatch', marker='<', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S11 (mag)')
        ax.set_ylim([0, 0.25])
        ax.set_yticks(np.arange(0, 0.26, 0.05))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,1]
        val_metas = abs(DUT_cal_metas[:,1,0]).squeeze()
        val_metas_reflect_only = abs(DUT_cal_metas_reflect[:,1,0]).squeeze()  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)                    
        std = np.sqrt(get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                      + get_cov_component(val_metas, cal_lin.Slines_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise', marker='^', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length offset', marker='v', markevery=15, markersize=10)
        std = munc.get_stdunc(val_metas_reflect_only)
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect asymmetry', marker='>', markevery=15, markersize=10)
        std = np.sqrt(get_cov_component(val_metas, cal_lin.ereff_Gamma_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Line mismatch', marker='<', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S21 (mag)')
        ax.set_ylim([0, 0.1])
        ax.set_yticks(np.arange(0, 0.11, 0.02))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.97), 
                   loc='lower center', ncol=3, borderaxespad=0,
                   title = r"95% uncertainty budget due to uncertainty type"
                   )
    
    ## uncertainties due to each standard
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0,0]
        val_metas = munc.umath.real(cal_lin.ereff)
        val_metas_reflect_only = munc.umath.real(cal_lin_reflect.ereff)  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,0]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,0]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,0]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 1 (thru)', marker='^', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,1]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,1]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,1]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 2', marker='v', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,2]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,2]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,2]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 3', marker='>', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,3]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,3]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,3]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 4', marker='<', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,4]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,4]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,4]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 5', marker='d', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,5]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,5]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,5]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 6', marker='X', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                          + munc.get_stdunc(val_metas_reflect_only)**2 )
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect', marker='h', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([0, 0.4])
        ax.set_yticks(np.arange(0, 0.41, 0.1))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[0,1]
        val_metas = gamma2dbmm(cal_lin.gamma)
        val_metas_reflect_only = gamma2dbmm(cal_lin_reflect.gamma)  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,0]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,0]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,0]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 1 (thru)', marker='^', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,1]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,1]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,1]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 2', marker='v', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,2]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,2]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,2]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 3', marker='>', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,3]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,3]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,3]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 4', marker='<', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,4]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,4]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,4]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 5', marker='d', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,5]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,5]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,5]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 6', marker='X', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                          + munc.get_stdunc(val_metas_reflect_only)**2 )
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect', marker='h', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 0.2])
        ax.set_yticks(np.arange(0, 0.21, 0.05))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,0]
        val_metas = abs(DUT_cal_metas[:,0,0]).squeeze()
        val_metas_reflect_only = abs(DUT_cal_metas_reflect[:,0,0]).squeeze()  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,0]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,0]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,0]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 1 (thru)', marker='^', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,1]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,1]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,1]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 2', marker='v', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,2]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,2]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,2]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 3', marker='>', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,3]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,3]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,3]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 4', marker='<', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,4]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,4]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,4]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 5', marker='d', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,5]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,5]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,5]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 6', marker='X', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                          + munc.get_stdunc(val_metas_reflect_only)**2 )
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect', marker='h', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S11 (mag)')
        ax.set_ylim([0, 0.25])
        ax.set_yticks(np.arange(0, 0.26, 0.05))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,1]
        val_metas = abs(DUT_cal_metas[:,1,0]).squeeze()
        val_metas_reflect_only = abs(DUT_cal_metas_reflect[:,1,0]).squeeze()  # to get around metas error 
        std = munc.get_stdunc(val_metas)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall', marker='o', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,0]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,0]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,0]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 1 (thru)', marker='^', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,1]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,1]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,1]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 2', marker='v', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,2]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,2]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,2]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 3', marker='>', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,3]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,3]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,3]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 4', marker='<', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,4]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,4]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,4]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 5', marker='d', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Slines_metas[:,5]) \
                      + get_cov_component(val_metas, cal_lin.lengths_metas[:,5]) \
                          + get_cov_component(val_metas, cal_lin.ereff_Gamma_metas[:,5]) )
        ax.plot(f*1e-9, std*k, lw=2, label='Line 6', marker='X', markevery=15, markersize=10)
        std = np.sqrt( get_cov_component(val_metas, cal_lin.Sreflect_metas) \
                          + munc.get_stdunc(val_metas_reflect_only)**2 )
        ax.plot(f*1e-9, std*k, lw=2, label='Reflect', marker='h', markevery=15, markersize=10)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S21 (mag)')
        ax.set_ylim([0, 0.1])
        ax.set_yticks(np.arange(0, 0.11, 0.02))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.97), 
                   loc='lower center', ncol=4, borderaxespad=0,
                   title = r"95% uncertainty budget due to cal standards"
                   )
    
    plt.show()
    
    # EOF