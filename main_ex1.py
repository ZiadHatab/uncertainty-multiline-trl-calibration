"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Example of on-wafer mTRL calibration on MPI ISS up to 150GHz 
"""

import os  
import skrf as rf      # for RF stuff
import numpy as np
import matplotlib.pyplot as plt   # for plotting
import metas_unclib as munc
munc.use_linprop()

# my code
from umTRL import umTRL

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
    
# main script
if __name__ == '__main__':
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    alpha2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))
    ualpha2dbmm = lambda x: mag2db(munc.umath.exp(munc.umath.real(x)*1e-3))
    
    # load the measurements
    # files' path are reference to script's path
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\s2p_MPI_ISS\\'
    
    # switch terms
    gamma_f = rf.Network(s2p_path + 'VNA_switch_term.s2p').s21
    gamma_r = rf.Network(s2p_path + 'VNA_switch_term.s2p').s12
    
    # Calibration standards
    L1    = rf.Network(s2p_path + 'MPI_line_0200u.s2p')
    L2    = rf.Network(s2p_path + 'MPI_line_0450u.s2p')
    L3    = rf.Network(s2p_path + 'MPI_line_0900u.s2p')
    L4    = rf.Network(s2p_path + 'MPI_line_1800u.s2p')
    L5    = rf.Network(s2p_path + 'MPI_line_3500u.s2p')
    L6    = rf.Network(s2p_path + 'MPI_line_5250u.s2p') # use as verification line
    SHORT = rf.Network(s2p_path + 'MPI_short.s2p')
    
    f = L1.frequency.f
    
    lines = [L1, 
             L2, 
             L3, 
             L4,
             L5,
             #L6
             ]
    line_lengths = [200e-6, 
                    450e-6, 
                    900e-6, 
                    1800e-6,
                    3500e-6,
                    #5250e-6
                    ]
    
    reflect = SHORT
    reflect_est = -1
    reflect_offset = 0
    ereff_est = 5+0j
    
    # uncertainties need to be defined as variance
    # the below values do not represent the actual uncertainties of the ISS nor 
    # the VNA. They are just numbers I estimated to showcase how the software works.
    
    sigma     = 0.002 # S-parameters iid AWGN
    uSlines   = sigma**2 # measured lines
    uSreflect = sigma**2 # measured reflect 
    usw       = sigma**2 # measured switch-terms
    ulengths  = (0.02e-3)**2  # uncertainty in length
    ureflect  = np.array([0.01, 0])**2  # uncertainty of the reflect standard
    uereff_Gamma = np.array([0.05, 0.5e-4, 0.002, 1.5e-7])**2 # mismatch uncertainty

    cal = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, switch_term=[gamma_f,gamma_r],
               uSlines=uSlines, ulengths=ulengths, uSreflect=uSreflect, 
               ureflect=ureflect, uereff_Gamma=uereff_Gamma,
               uswitch_term=usw
               )
    cal.run_umTRL()
    
    dut_embed = L6
    dut_cal, dut_cov = cal.apply_cal(dut_embed)
    S_cal = np.array([munc.ucomplexarray(x, covariance=y) for x,y 
                       in zip(dut_cal.s.squeeze(), dut_cov)])
    
    # effective permittivity and loss
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3))        
        fig.set_dpi(600)
        fig.tight_layout(pad=1.5)
        ax = axs[0]
        plot_with_unc(ax, f, munc.get_value(cal.ereff).real, 
                  munc.get_stdunc(cal.ereff).real, 
                  label='Linear uncertainty', marker='^', markevery=60, markersize=10)
        ax.set_ylabel('Effective permittivity')
        ax.set_ylim([4.5, 6])
        ax.set_yticks(np.arange(4.5, 6.1, 0.5))
        
        ax = axs[1]
        losses_dbmm = ualpha2dbmm(cal.gamma)
        plot_with_unc(ax, f, munc.get_value(losses_dbmm).real, 
                      munc.get_stdunc(losses_dbmm).real, 
                      label='Linear uncertainty', marker='^', 
                      markevery=60, markersize=10)
        ax.set_ylabel('Losses (dB/mm)')
        ax.set_ylim([0, 1])
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        plt.suptitle("Effective permittivity and loss per-unit-length", 
             verticalalignment='bottom')#.set_y(1.1)
    
    # calibrated dut
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3))        
        fig.set_dpi(600)
        fig.tight_layout(pad=1.5)
        
        ax = axs[0]
        S11abs = abs(S_cal[:,0,0])
        plot_with_unc(ax, f, munc.get_value(S11abs).real, 
                  munc.get_stdunc(S11abs).real, 
                  label='Linear uncertainty', marker='^', markevery=60, markersize=10)
        ax.set_ylabel('S11 (mag)')
        ax.set_ylim([-0.02, 0.06])
        ax.set_yticks(np.arange(-0.02, 0.065, 0.02))
        
        ax = axs[1]
        S21abs = abs(S_cal[:,1,0])
        plot_with_unc(ax, f, munc.get_value(S21abs).real, 
                      munc.get_stdunc(S21abs).real, 
                      label='Linear uncertainty', marker='^', 
                      markevery=60, markersize=10)
        ax.set_ylabel('S21 (mag)')
        ax.set_ylim([0.6, 1])
        ax.set_yticks(np.arange(0.6, 1.05, 0.1))
        plt.suptitle("Calibrated verification line (5.25 mm)", 
             verticalalignment='bottom')#.set_y(1.1)
    
    plt.show()
    
# EOF