"""Penalty for f' and f" violating Kramers Kronig relations"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from scipy.interpolate import CubicSpline,interp1d
from scipy.signal.windows import get_window

INTERP_FUNC = interp1d
                              
def parse_data(path, remove_first_line=False):
    """Load and parse input."""
    data_input=pathlib.Path(path).read_text().rstrip()
    lines = data_input.split('\n')
    if remove_first_line:
        start_ind=1
    else:
        start_ind=0
    sf = np.array([[float(p) for p in line.split()] for line in lines[start_ind:]])
    return(sf)
  

def get_f_dp(energy, f_p,
             padn=5000,
             Z=26, # atomic number
             include_Z_term=False,
             ):
    
    """Derive f" (f1) from f' (f2) """
    energy,f_dp_pred = get_f_p(energy, -f_p, padn=padn,
                               Z=Z, # atomic number
                               include_Z_term=include_Z_term,
                               )
    return(energy,f_dp_pred)


def fftkk(f_dp, energy, padn=5000):
    """ Code adapted from Sherrell thesis """
    denergy = energy[1:]-energy[:-1]
    dE = energy[1]-energy[0]
    if np.any(denergy-denergy[0]): # nonuniform spacing
        uniform_mesh = np.arange(energy[0],energy[-1],dE)
        interp = INTERP_FUNC(energy, f_dp)
        F=interp(uniform_mesh) # interpolated f_dp
        energy=uniform_mesh
    else:
        F = f_dp   

    # sin padding as used in Sherrell thesis
    S = np.sin(np.linspace(-np.pi / 2, np.pi / 2, padn)) * 0.5 + 0.5
    Fin = np.hstack((F[0] * np.ones_like(S),
                      F,
                      F[-1] * np.ones_like((1 - S))))  
    
    
    Hz = np.fft.fft(Fin)
    H = np.zeros_like(Hz)
    H[0]=0
    H[1:(len(H)//2)]=-1
    H[(len(H)//2):]=1
    H = H*1j
   
    dF = np.fft.ifft(1j*H*Hz+Hz) #.reshape((1, npts))
    return dF[padn:-padn].imag

def get_f_p(energy, f_dp, padn=5000,
            Z = 26, # atomic number
            relativistic_corr=False,
            ):
    """Derive f' (f2) from f" (f1) """
    
    denergy = energy[1:]-energy[:-1]
    dE = energy[1]-energy[0]
    if np.any(denergy-denergy[0]): # nonuniform spacing
        uniform_mesh = np.arange(energy[0],energy[-1],dE)
        interp = INTERP_FUNC(energy, f_dp)
        F=interp(uniform_mesh) # interpolated f_dp
        energy=uniform_mesh
    else:
        F = f_dp   

    # sin padding as used in Sherrell thesis
    S = np.sin(np.linspace(-np.pi / 2, np.pi / 2, padn)) * 0.5 + 0.5
    Fin = np.hstack((F[0] * np.ones_like(S),
                      F,
                      F[-1] * np.ones_like((1 - S))))  

    if padn==0:
        f_p_pred = np.imag(hilbert(Fin, N=None, axis=- 1))
    else:
        f_p_pred = np.imag(hilbert(Fin, N=None, axis=- 1))[padn:-padn]
    
    if relativistic_corr:
        Z_star = Z - (Z/82.5)**2.37
    else:
        Z_star=0
    f_p_pred = Z_star + f_p_pred
    
    return(energy,f_p_pred)


def create_window(padn):
    
    
def penalty(energy, f_p, f_dp, padn=5000, plot=True, start_plot=0,
            end_plot=30000, relativistic_corr=False):
    
    energy_interp,f_p_pred = get_f_p(energy, f_dp, padn=padn,
                                     relativistic_corr=relativistic_corr)
    
    # interpolation in case of nonuniform energy
    f_p_interp = INTERP_FUNC(energy, f_p)(energy_interp)
    
    start_ind = np.argmin(np.abs(energy_interp-start_plot))
    end_ind = np.argmin(np.abs(energy_interp-end_plot))
    if plot:
        plt.figure()
        plt.plot(energy_interp[start_ind:end_ind], f_p_interp[start_ind:end_ind], label='actual')
        plt.plot(energy_interp[start_ind:end_ind], f_p_pred[start_ind:end_ind], label='predicted')
        if not(relativistic_corr):
            plt.ylim([-8,3])
        plt.legend()
    mse = np.mean((f_p_interp - f_p_pred)**2)
    return(mse)
    
    
if __name__ == "__main__":
    start_plot=7070
    end_plot=7170
    
    path='sample_data/Fe.nff' # Henke data
    sf = parse_data(path, remove_first_line=True)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]

    mse = penalty(energy, f_p, f_dp, padn=0,
                  start_plot=start_plot, end_plot=end_plot,
                  relativistic_corr=True)
    print(mse)
    
    path='sample_data/pf-rd-red_fftkk.out'
    sf = parse_data(path)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]

    
    mse = penalty(energy, f_p, f_dp, 
                  start_plot=start_plot, end_plot=end_plot,
                  padn=5000)
    
    print(mse)
    
    path = 'sample_data/pf-rd-ox_fftkk.out'
    sf = parse_data(path)
    energy = sf[:,0][:-1]
    f_p = sf[:,1][:-1]
    f_dp = sf[:,2][:-1]
    
    
    mse = penalty(energy, f_p, f_dp, 
                  start_plot=start_plot, end_plot=end_plot,
                  padn=5000)



    print(mse)
    
    f_p_sherrell = fftkk(f_dp, energy,padn=5000)
    start_ind = np.argmin(np.abs(energy-start_plot))
    end_ind = np.argmin(np.abs(energy-end_plot))
    
    plt.figure()
    plt.plot(energy[start_ind:end_ind], f_p[start_ind:end_ind], label='actual')
    plt.plot(energy[start_ind:end_ind], f_p_sherrell[start_ind:end_ind], label='predicted')
    plt.ylim([-8,3])
    plt.legend()
