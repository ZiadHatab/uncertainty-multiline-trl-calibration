"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Multiline TRL Linear Uncertainty (LU). Demonstrating the impact of different uncertainty 
types on the final output. Simulated data used here.
"""

import skrf as rf      # for RF stuff
from skrf.media import Coaxial, CPW
import numpy as np
import matplotlib.pyplot as plt   # for plotting
import metas_unclib as munc
munc.use_linprop()

# my code
from umTRL import umTRL
import timeit

class PlotSettings:
    # https://matplotlib.org/3.3.3/tutorials/introductory/customizing.html
    def __init__(self, font_size=10, latex=False): 
        self.font_size = font_size 
        self.latex = latex
    def __enter__(self):
        plt.style.use('seaborn-paper')
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

def plot_with_unc(ax, f, x, ux, label='', title='', f_scale=1e-9, k=2, 
                  markevery=None, marker=None,markersize=None, alpha=0.3):
    ax.plot(f*f_scale, x, lw=2, label=label, markevery=markevery, 
            marker=marker, markersize=markersize)
    ax.fill_between(f*f_scale, x-ux*k, x+ux*k, alpha=alpha)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_xticks(np.arange(0,151,30))
    ax.set_xlim([0, max(f*f_scale)])
    return None

def plot_with_unc2(ax, f, x, ux, label='', title='', f_scale=1e-9, k=2, 
                  markevery=None, marker=None,markersize=None):
    line = ax.plot(f*f_scale, x, lw=2, label=label, markevery=markevery, 
            marker=marker, markersize=markersize)
    ax.plot(f*f_scale, x-ux*k, linestyle=(0, (5, 5)), lw=1.5, color=line[0].get_color())
    ax.plot(f*f_scale, x+ux*k, linestyle=(0, (5, 5)), lw=1.5, color=line[0].get_color())
    ax.set_xlabel('Frequency (GHz)')
    ax.set_xticks(np.arange(0,151,30))
    ax.set_xlim([0, max(f*f_scale)])
    return None

def S2T(S):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T/S[1,0]

def T2S(T):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S/T[1,1]

def add_white_noise(NW, sigma=0.01):
    # add white noise to a network's S-paramters
    NW_new = NW.copy()
    noise = (np.random.standard_normal(NW_new.s.shape) 
             + 1j*np.random.standard_normal(NW_new.s.shape))*sigma
    NW_new.s = NW_new.s + noise
    return NW_new

def Qnm(Zn, Zm):
    # Impedance transformer in T-paramters from on Eqs. (86) and (87) in
    # R. Marks and D. Williams, "A general waveguide circuit theory," 
    # Journal of Research (NIST JRES), National Institute of Standards and Technology,
    # Gaithersburg, MD, no. 97, 1992.
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914227/
    Gnm = (Zm-Zn)/(Zm+Zn)
    return np.sqrt(Zn.real/Zm.real*(Zm/Zn).conjugate())/np.sqrt(1-Gnm**2)*np.array([[1, Gnm],[Gnm, 1]])
    
def TL(l, cpw, Z01=None, Z02=None):
    # create mismatched transmission line from skrf cpw object
    # the line function from skrf is not fully correct. it has a 180deg offset 
    # in the S11 and S22 (I will let them know about it).
    N = len(cpw.Z0)  # number of frequency points
    Z01 = cpw.Z0 if Z01 is None else np.atleast_1d(Z01)*np.ones(N)
    Z02 = Z01 if Z02 is None else np.atleast_1d(Z02)*np.ones(N)
    S = []
    for g,zc,z01,z02 in zip(cpw.gamma, cpw.Z0, Z01, Z02):
        T = Qnm(z01,zc)@np.diag([np.exp(-l*g), np.exp(l*g)])@Qnm(zc,z02)
        S.append(T2S(T))
    
    return rf.Network(s=np.array(S), frequency=cpw.frequency, name='From T2S')

def ideal_sym_DUT(freq):
    # ideal lossless, symmetrical, equal reflection and transmission network
    s = 1/np.sqrt(2)
    S = np.array([[s, s], [s, s]])
    return rf.Network(s=np.tile(S, (len(freq.f), 1,1)), frequency=freq)

if __name__=='__main__':
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    alpha2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))
    ualpha2dbmm = lambda x: mag2db(munc.umath.exp(munc.umath.real(x)*1e-3))
    
    # define frequency range
    freq = rf.F(0.5, 150, 300, unit='GHz')
    f = freq.f
    
    # 1.0 mm coaxial media for calibration error boxes
    coax1mm = Coaxial(freq, z0=50, Dint=0.44e-3, Dout=1.0e-3, sigma=1e8)
    A = coax1mm.line(1, 'm', z0=49, name='X', embed=True) # left
    B = coax1mm.line(1.01, 'm', z0=51, name='Y', embed=True) # right
    
    # CPW media used for the calibration standards
    cpw_ori = CPW(freq, w=40e-6, s=25e-6, ep_r=12*(1-0.001j), t=5e-6, rho=2e-8)
        
    # line standards
    line_lengths = [0, 0.5e-3, 1.75e-3, 3.5e-3, 3.75e-3, 4.5e-3, 6e-3]
    lines = [A**TL(l, cpw_ori)**B for l in line_lengths]
    
    # reflect standard
    reflect_est = -1
    reflect_offset = 0
    SHORT = rf.two_port_reflect( cpw_ori.delay_short(reflect_offset, 'm') )
    reflect = A**SHORT**B
    
    # embedded DUT
    dut = ideal_sym_DUT(freq)
    dut_embed = A**dut**B
    
    ereff_est = 6+0j
    
    # Uncertainties need to be defined as variance/covariance
    sigma     = 0.004 # S-parameters iid AWGN
    uSlines   = sigma**2 # [sigma**2 for l in line_lengths]
    uSreflect = sigma**2
    ulengths  = (0.02e-3)**2
    sig_ref   = 0.05
    ureflect  = sig_ref**2
    
    # Estimate the covariance of the ereff and mismatch of the cpw
    # you could just take the equations and compute it manually, but i'm too lazy to that!
    sigma_er = 0.1
    M = 2000
    Z0_mc = []
    ereff_mc = []
    for m in range(M):
        ep_r = 12 + np.random.randn()*sigma_er
        cpw = CPW(freq, w=40e-6, s=25e-6, ep_r=ep_r*(1-0.001j), t=5e-6, rho=2e-8)
        Z0_mc.append(cpw.Z0[0])
        ereff_mc.append(cpw.ep_reff[0])
    Z0_mc = np.array(Z0_mc)
    ereff_mc = np.array(ereff_mc)
    G_mc = (Z0_mc-cpw_ori.Z0[0])/(Z0_mc+cpw_ori.Z0[0])
    uereff_Gamma = np.cov([ereff_mc.real, ereff_mc.imag, G_mc.real, G_mc.imag])
    # uereff_Gamma = np.array([0.05, 0.5e-4, 0.002, 1.5e-7])**2
    
    # linear uncertainty
    cal_all = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, switch_term=None,
               uSlines=uSlines, ulengths=ulengths, uSreflect=uSreflect, 
               ureflect=ureflect, uereff_Gamma=uereff_Gamma,
               )
    cal_all.run_umTRL()

    cal_awgn = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, switch_term=None,
               uSlines=uSlines, ulengths=ulengths*0, uSreflect=uSreflect, 
               ureflect=ureflect*0, uereff_Gamma=uereff_Gamma*0,
               )
    cal_awgn.run_umTRL()

    cal_len = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, switch_term=None,
               uSlines=uSlines*0, ulengths=ulengths, uSreflect=uSreflect*0, 
               ureflect=ureflect*0, uereff_Gamma=uereff_Gamma*0,
               )
    cal_len.run_umTRL()
    
    cal_refc = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, switch_term=None,
               uSlines=uSlines*0, ulengths=ulengths*0, uSreflect=uSreflect*0, 
               ureflect=ureflect, uereff_Gamma=uereff_Gamma*0,
               )
    cal_refc.run_umTRL()
    
    cal_mism = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, switch_term=None,
               uSlines=uSlines*0, ulengths=ulengths*0, uSreflect=uSreflect*0, 
               ureflect=ureflect*0, uereff_Gamma=uereff_Gamma,
               )
    cal_mism.run_umTRL()
    
    def get_uncS(nw, cov):
        return np.array([munc.ucomplexarray(x, covariance=y) for x,y 
                           in zip(nw.s.squeeze(), cov)])
        
    dut_cal, cov = cal_all.apply_cal(dut_embed)
    dut_cal_all = get_uncS(dut_cal, cov)
    
    dut_cal, cov = cal_awgn.apply_cal(dut_embed)
    dut_cal_awgn = get_uncS(dut_cal, cov)
    
    dut_cal, cov = cal_len.apply_cal(dut_embed)
    dut_cal_len = get_uncS(dut_cal, cov)
    
    dut_cal, cov = cal_refc.apply_cal(dut_embed)
    dut_cal_refc = get_uncS(dut_cal, cov)
    
    dut_cal, cov = cal_mism.apply_cal(dut_embed)
    dut_cal_mism = get_uncS(dut_cal, cov)

    ## PLOTS
    k = 2
    # dielectric constant and losses
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2.5)
        ax = axs[0]
        ax.plot(f*1e-9, munc.get_stdunc(cal_all.ereff).real*k, lw=2,
                  label='All', marker='^', markevery=20, markersize=10)
        ax.plot(f*1e-9, munc.get_stdunc(cal_awgn.ereff).real*k, lw=2,
                  label='AWGN', marker='>', markevery=20, markersize=10)
        ax.plot(f*1e-9, munc.get_stdunc(cal_len.ereff).real*k, lw=2,
                  label='Length unc.', marker='<', markevery=20, markersize=10)
        ax.plot(f*1e-9, munc.get_stdunc(cal_refc.ereff).real*k, lw=2,
                  label='Reflect unc.', marker='v', markevery=20, markersize=10)
        ax.plot(f*1e-9, munc.get_stdunc(cal_mism.ereff).real*k, lw=2,
                  label='Mismatch unc.', marker='o', markevery=20, markersize=10)
        ax.set_ylabel('Effective permittivity')
        ax.set_ylim([0, 0.4])
        ax.set_yticks(np.arange(0, 0.45, 0.1))
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,151,30))
        ax.set_xlim([0, max(f*1e-9)])
        
        ax = axs[1]
        losses_dbmm = ualpha2dbmm(cal_all.gamma)
        ax.plot(f*1e-9, munc.get_stdunc(losses_dbmm).real*k, lw=2,
                      label='All', marker='^', markevery=20, markersize=10)
        losses_dbmm = ualpha2dbmm(cal_awgn.gamma)
        ax.plot(f*1e-9, munc.get_stdunc(losses_dbmm).real*k, lw=2,
                  label='AWGN', marker='>', markevery=20, markersize=10)
        losses_dbmm = ualpha2dbmm(cal_len.gamma)
        ax.plot(f*1e-9, munc.get_stdunc(losses_dbmm).real*k, lw=2,
                  label='Length unc.', marker='<', markevery=20, markersize=10)
        losses_dbmm = ualpha2dbmm(cal_refc.gamma)
        ax.plot(f*1e-9, munc.get_stdunc(losses_dbmm).real*k, lw=2,
                  label='Reflect unc.', marker='v', markevery=20, markersize=10)
        losses_dbmm = ualpha2dbmm(cal_mism.gamma)
        ax.plot(f*1e-9, munc.get_stdunc(losses_dbmm).real*k, lw=2,
                  label='Mismatch unc.', marker='o', markevery=20, markersize=10)
        ax.set_ylabel('Losses (dB/mm)')
        ax.set_ylim([0, 0.06])
        ax.set_yticks(np.arange(0, 0.065, 0.02))
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,151,30))
        ax.set_xlim([0, max(f*1e-9)])
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.9), 
                   loc='lower center', ncol=3, borderaxespad=0)
        plt.suptitle("95% expanded uncertainty of effective permittivity and losses per-unit-length", 
             verticalalignment='bottom').set_y(1.15)
    
    # calibrated DUT S11 and S21
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2.5)
        ax = axs[0]
        S11abs = abs(dut_cal_all[:,0,0])
        ax.plot(f*1e-9, munc.get_stdunc(S11abs).real*k, lw=2,
                      label='All', marker='^', markevery=20, markersize=10)
        S11abs = abs(dut_cal_awgn[:,0,0])
        ax.plot(f*1e-9, munc.get_stdunc(S11abs).real*k, lw=2,
                  label='AWGN', marker='>', markevery=20, markersize=10)
        S11abs = abs(dut_cal_len[:,0,0])
        ax.plot(f*1e-9, munc.get_stdunc(S11abs).real*k, lw=2,
                  label='Length unc.', marker='<', markevery=20, markersize=10)
        S11abs = abs(dut_cal_refc[:,0,0])
        ax.plot(f*1e-9, munc.get_stdunc(S11abs).real*k, lw=2,
                  label='Reflect unc.', marker='v', markevery=20, markersize=10)
        S11abs = abs(dut_cal_mism[:,0,0])
        ax.plot(f*1e-9, munc.get_stdunc(S11abs).real*k, lw=2,
                  label='Mismatch unc.', marker='o', markevery=20, markersize=10)
        ax.set_ylabel('S11 (mag)')
        ax.set_ylim([0, 0.08])
        ax.set_yticks(np.arange(0, 0.081, 0.02))
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,151,30))
        ax.set_xlim([0, max(f*1e-9)])
        
        ax = axs[1]
        S21abs = abs(dut_cal_all[:,1,0])
        ax.plot(f*1e-9, munc.get_stdunc(S21abs).real*k, lw=2,
                      label='All', marker='^', markevery=20, markersize=10)
        S21abs = abs(dut_cal_awgn[:,1,0])
        ax.plot(f*1e-9, munc.get_stdunc(S21abs).real*k, lw=2,
                  label='AWGN', marker='>', markevery=20, markersize=10)
        S21abs = abs(dut_cal_len[:,1,0])
        ax.plot(f*1e-9, munc.get_stdunc(S21abs).real*k, lw=2,
                  label='Length unc.', marker='<', markevery=20, markersize=10)
        S21abs = abs(dut_cal_refc[:,1,0])
        ax.plot(f*1e-9, munc.get_stdunc(S21abs).real*k, lw=2,
                  label='Reflect unc.', marker='v', markevery=20, markersize=10)
        S21abs = abs(dut_cal_mism[:,1,0])
        ax.plot(f*1e-9, munc.get_stdunc(S21abs).real*k, lw=2,
                  label='Mismatch unc.', marker='o', markevery=20, markersize=10)
        ax.set_ylabel('S21 (mag)')
        ax.set_ylim([0, 0.06])
        ax.set_yticks(np.arange(0, 0.061, 0.02))
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,151,30))
        ax.set_xlim([0, max(f*1e-9)])
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.9), 
                   loc='lower center', ncol=3, borderaxespad=0)
        plt.suptitle("95% expanded uncertainty of calibrated DUT", 
             verticalalignment='bottom').set_y(1.15)
    
    plt.show()
    
