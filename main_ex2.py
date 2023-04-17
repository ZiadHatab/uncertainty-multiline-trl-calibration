"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Example of mTRL uncertainly propagation using multi-sweep measurements.
The mean value of the sweep is used in the calibration.
The uncertainty due to the VNA was determined from the multi-sweep as sample covariance.
The uncertainty due to the standards are estimated using CPW model.
"""
import os
import zipfile

# pip install numpy matplotlib scikit-rf metas_unclib scipy -U
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import norm  # for fitting 2d Gaussian hist
import metas_unclib as munc
munc.use_linprop()

# umTRL.py and cpw.py should be in same folder 
from umTRL import umTRL
from cpw import CPW   # model of CPW

def read_waves_to_S_from_zip(zipfile_full_dir, file_name_contain):
    # read wave parameter files and convert to S-parameters (from a zip file)
    with zipfile.ZipFile(zipfile_full_dir, mode="r") as archive:
        netwks = rf.read_zipped_touchstones(archive)
        A = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_A' in key])
        B = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_B' in key])    
    freq = A[0].frequency
    S = rf.NetworkSet( [rf.Network(s=b.s@np.linalg.inv(a.s), frequency=freq) for a,b in zip(A,B)] )
    return S.mean_s, S.cov(), np.array([s.s for s in S])

def plot_2d_hist(fig,x,y,bins=30):
    # adapted from: 
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    # segment the figure
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.08, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelbottom=False, bottom=False, top=True, labeltop=True)
    
    # the scatter plot:
    ax.scatter(x, y, alpha=0.8)
    ax_histx.hist(x, bins=bins, density=True, alpha=0.8)    
    ax_histy.hist(y, bins=bins, orientation='horizontal', density=True, alpha=0.8)
    
    return ax, ax_histx, ax_histy

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    ## from here: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
        
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

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
    # these data are already corrected with switch term effects
    L1, L1_cov, L1S = read_waves_to_S_from_zip(path + f'{file_name}_thru.zip', f'{file_name}_thru')
    L2, L2_cov, L2S = read_waves_to_S_from_zip(path + f'{file_name}_line01.zip', f'{file_name}_line01')
    L3, L3_cov, L3S = read_waves_to_S_from_zip(path + f'{file_name}_line02.zip', f'{file_name}_line02')
    L4, L4_cov, L4S = read_waves_to_S_from_zip(path + f'{file_name}_line03.zip', f'{file_name}_line03')
    L5, L5_cov, L5S = read_waves_to_S_from_zip(path + f'{file_name}_line04.zip', f'{file_name}_line04')
    L6, L6_cov, L6S = read_waves_to_S_from_zip(path + f'{file_name}_line05.zip', f'{file_name}_line05')
    OPEN, OPEN_cov, OPENS = read_waves_to_S_from_zip(path + f'{file_name}_open.zip', f'{file_name}_open')
    f = L1.frequency.f  # frequency axis
    
    # plot 2D hist diagrams
    tiks = [ (np.arange(0.2,0.241, 0.02), np.arange(0.04, 0.122, 0.04), np.arange(0,101,100)),
             (np.arange(-0.07, -0.049, 0.01), np.arange(-0.06, -0.039, 0.01), np.arange(0,201,200))]
    f_inx = 119
    for tik, inx in zip(tiks,[(1,1),(0,1)]):
        x = L4S[:,f_inx,inx[0],inx[1]].real
        y = L4S[:,f_inx,inx[0],inx[1]].imag
        with PlotSettings(14):
            fig = plt.figure(figsize=(5,5))
            fig.set_dpi(600)                   
            ax,axx,axy = plot_2d_hist(fig,x,y)
            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
        
            # Fit a normal distribution to the data:
            # x data
            mux, stdx = norm.fit(x)
            xmin, xmax = ax.get_xlim()
            xx = np.linspace(xmin, xmax, len(x))
            p = norm.pdf(xx, mux, stdx)
            axx.plot(xx, p, 'r', lw=2)
            # y data
            muy, stdy = norm.fit(y)
            ymin, ymax = ax.get_ylim()
            yy = np.linspace(ymin, ymax, len(y))
            p = norm.pdf(yy, muy, stdy)
            axy.plot(p, yy, 'r', lw=2)
            
            ax.plot([],[], lw=2, color='red')
            ax.scatter(mux, muy, c='red', s=3)
            for n in [1,2,3]:
                confidence_ellipse(x,y,ax,n_std=n, edgecolor='red', lw=2)
            mu = L4S.mean(axis=0)
            ax.legend(['Measured','Estimated'])
            plt.suptitle(f'Histogram of S{inx[0]+1}{inx[1]+1} at {f[f_inx]*1e-9:.0f}GHz', 
                 verticalalignment='bottom').set_y(0.94)
    
    # mTRL definition
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6, 3500e-6, 5250e-6]
    reflect = OPEN
    reflect_est = 1
    reflect_offset = -0.1e-3
    ereff_est = 5.45-0.0001j
    
    # CPW model parameters (used for estimation mismatch uncertainty)
    w, s, wg, t = 49.1e-6, 25.5e-6, 273.3e-6, 4.9e-6
    Dk = 9.9
    Df = 0.0
    sig = 4.11e7  # conductivity of Gold
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sig)
    cpw.update_jac() # compute the Jacobian of the cpw with respect to its inputs
        
    ## define uncertainties
    # Noise uncertainties
    uSlines   = np.array([L1_cov, L2_cov, L3_cov, L4_cov, L5_cov, L6_cov]) # measured lines
    uSreflect = OPEN_cov # measured reflect 
    
    # length uncertainties
    l_std = 40e-6  # for the line
    ulengths  = l_std**2  # the umTRL code will automatically repeat it for all lines
    l_open_std = 40e-6 # uncertainty in length used for the reflect
    
    # cross-section uncertainties
    w_std   = 2.55e-6
    s_std   = 2.55e-6
    wg_std  = 2.55e-6
    t_std   = 0.49e-6
    Dk_std  = 0.2
    Df_std  = 0
    sig_std = sig*0.1

    # line mismatch uncertainty
    U = np.diag([w_std,s_std,wg_std,t_std,Dk_std,Df_std,sig_std])**2
    uereff_Gamma_i = np.array([ np.vstack((x,y)).dot(U).dot(np.vstack((x,y)).T) for x,y in zip(cpw.jac_ereff,cpw.jac_Gamma)])
    uereff_Gamma   = [uereff_Gamma_i]*len(lines) # repeat for all lines
    
    # open asymmetry
    # the uncertainty is computed analyically as an offset asymmetry between the ports
    diff_open = lambda g,l: -2*g*np.exp(-2*g*l)
    ureflect     = np.array([ np.array([[diff_open(g,reflect_offset).real],[diff_open(g,reflect_offset).imag]]).dot(
        np.array([[diff_open(g,reflect_offset).real],[diff_open(g,reflect_offset).imag]]).T)*l_open_std**2 for g in cpw.gamma])
    
    # mTRL with linear uncertainty evaluation
    cal = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est,
               uSlines=uSlines, uSreflect=uSreflect,
               ulengths=ulengths,
               ureflect=ureflect, uereff_Gamma=uereff_Gamma)
    cal.run_umTRL() # run mTRL with linear uncertainty propagation
    
    dut_embed = L6
    dut_cal, dut_cal_S = cal.apply_cal(dut_embed)
    
    # plot data and uncertainty
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
        ax.set_ylim([0, 1.5])
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
        ax.set_ylim([-0.06, 0.1])
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