"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Example of mTRL uncertainly propagation using measurements with switch-terms.
The uncertainty values given here are just for illustration and are not accurate representation of the measurement.
"""

import os

# pip install numpy matplotlib scikit-rf metas_unclib -U
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt   # for plotting
import metas_unclib as munc
munc.use_linprop()

# umTRL.py should be in same folder as this script
from umTRL import umTRL

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

if __name__ == '__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    
    # load the measurements
    # files' path are reference to script's path
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\MPI_ISS_measurements\\'
    
    # switch terms
    gamma_f = rf.Network(s2p_path + 'VNA_switch_term.s2p').s21
    gamma_r = rf.Network(s2p_path + 'VNA_switch_term.s2p').s12
    
    # Calibration standards
    L1    = rf.Network(s2p_path + 'MPI_line_0200u.s2p')
    L2    = rf.Network(s2p_path + 'MPI_line_0450u.s2p')
    L3    = rf.Network(s2p_path + 'MPI_line_0900u.s2p')
    L4    = rf.Network(s2p_path + 'MPI_line_1800u.s2p')
    L5    = rf.Network(s2p_path + 'MPI_line_3500u.s2p')
    L6    = rf.Network(s2p_path + 'MPI_line_5250u.s2p') # use as DUT
    SHORT = rf.Network(s2p_path + 'MPI_short.s2p')
    f = L1.frequency.f
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6, 3500e-6, 5250e-6]
    reflect = SHORT
    reflect_est = -1
    reflect_offset = -100e-6
    ereff_est = 5+0j
    
    # Uncertainties need to be defined as variance/covariance.
    # The below values do not represent the actual uncertainties of the ISS nor 
    # the VNA. They are just numbers I came up with to showcase how the code works.
    sigma     = 0.002 # S-parameters iid Gaussian noise
    uSlines   = sigma**2 # line measurements uncertainty (all lines have same uncertainty)
    uSreflect = sigma**2 # reflect measurements uncertainty
    usw       = sigma**2 # switch-terms measurements uncertainty
    ulengths  = (0.02e-3)**2  # uncertainty in length (all lines have same uncertainty)
    ureflect  = np.diag([0.01, 0])**2  # uncertainty of the reflect standard
    uereff_Gamma = np.diag([0.05, 0.5e-4, 0.002, 1.5e-7])**2 # mismatch uncertainty (all lines have same uncertainty)

    cal = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, switch_term=[gamma_f,gamma_r],
               uSlines=uSlines, ulengths=ulengths, uSreflect=uSreflect, 
               ureflect=ureflect, uereff_Gamma=uereff_Gamma,
               uswitch_term=usw)
    cal.run_umTRL()
    
    dut_embed = L6
    dut_cal, dut_cal_S = cal.apply_cal(dut_embed)
    
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,7))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0,0]
        mu  = munc.get_value(cal.ereff).real
        std = munc.get_stdunc(cal.ereff).real
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([4.5, 6.5])
        #ax.set_yticks(np.arange(4.5, 6.01, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[0,1]
        loss_dbmm_mTRL_model_lin = gamma2dbmm(cal.gamma)
        mu  = munc.get_value(loss_dbmm_mTRL_model_lin)
        std = munc.get_stdunc(loss_dbmm_mTRL_model_lin)
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 1])
        #ax.set_yticks(np.arange(0, 1.51, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,0]
        mu  = munc.get_value(abs(dut_cal_S[:,0,0])).squeeze()
        std = munc.get_stdunc(abs(dut_cal_S[:,0,0])).squeeze()
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S11 (mag)')
        ax.set_ylim([-0.04, 0.08])
        #ax.set_yticks(np.arange(.5, 0.91, 0.1))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1,1]
        mu  = munc.get_value(abs(dut_cal_S[:,1,0])).squeeze()
        std = munc.get_stdunc(abs(dut_cal_S[:,1,0])).squeeze()
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S21 (mag)')
        ax.set_ylim([0.5, 1])
        #ax.set_yticks(np.arange(.5, 0.91, 0.1))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()

        plt.suptitle(r"CPW parameters and calibrated DUT with 95% uncertainty bounds ($2\times\sigma$)", 
             verticalalignment='bottom').set_y(0.98)

    plt.show()
    
# EOF