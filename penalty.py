"""Penalty for f' and f" violating Kramers Kronig relations"""

import sys
import pathlib
import numpy as np

from scipy.signal import hilbert
from scipy.interpolate import CubicSpline,interp1d
import matplotlib.pyplot as plt

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
  

def get_f_dp(f_p):
    """Derive f" (f1) from f' (f2) """
    pass


def fftkk(dF_in, E, switch=1, grain=0.05):
    """ Code from Sherrell thesis """
    #Pad out the spectrum by 5000 points
    k = 5000
    E_dn = np.arange(E[0] - grain, E[0] - (k+1)*grain, -grain)
    E_dn = np.fliplr(E_dn.reshape(1, E_dn.shape[0])).reshape(E_dn.shape[0],)
    E_up = np.arange(E[-1] + grain, E[-1] + (k+1)*grain, grain)
    _E_ = np.hstack((E_dn, E, E_up))
    #Take edge of spectrum gently to zero using a quarter of a sine wave
    range = np.linspace(-np.pi/2, np.pi/2, k)
    dn = (np.sin(range) / 2) + 0.5
    up = 1 - dn
    dF_dn = dF_in[0] * dn
    dF_up = dF_in[-1] * up
    _dF_ = np.hstack((dF_dn, dF_in, dF_up))
    #This is based on fftkk.f by Graham George which
    #in turn is based on a paper.
    npts = _E_.shape[0]
    Hz = np.fft.fft(_dF_)
    mn = Hz[0]
    front = Hz[0:(len(Hz)//2)]
    tmp = front.copy()
    #I believe the next line is the magic.  Its the convolution with the signum or
    #how to bypass the difficult integration.
    tmp.real, tmp.imag = front.imag, front.real
    front = switch * tmp
    back = Hz[(len(Hz)//2):]
    tmp2 = back.copy()
    tmp2.real, tmp2.imag = back.imag, back.real
    back = -switch * tmp2
    new_Hz = np.hstack((mn, front, back))
    dF = np.fft.ifft(new_Hz).reshape((1, npts))
    dF_out = np.fliplr(dF).reshape((npts,))
    return dF_out[k:-k].real

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
        plt.legend()
    mse = np.mean((f_p_interp - f_p_pred)**2)
    return(mse)
    
    
if __name__ == "__main__":
    
    
    path='sample_data/Fe.nff' # Henke data
    sf = parse_data(path, remove_first_line=True)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]

    mse = penalty(energy, f_p, f_dp, padn=0,
                  start_plot=4000, end_plot=20000,
                  relativistic_corr=True)
    print(mse)
    
    path='sample_data/pf-rd-red_fftkk.out'
    sf = parse_data(path)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]

    
    mse = penalty(energy, f_p, f_dp, 
                  start_plot=4000, end_plot=20000,
                  padn=5000)
    
    print(mse)
    
    path = 'sample_data/pf-rd-ox_fftkk.out'
    sf = parse_data(path)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]
    
    
    mse = penalty(energy, f_p, f_dp, 
                  start_plot=4000, end_plot=20000,
                  padn=5000)

    print(mse)
