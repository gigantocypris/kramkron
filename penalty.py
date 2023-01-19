"""Penalty for f' and f" violating Kramers Kronig relations"""

import pathlib
import numpy as np

from scipy.signal import hilbert
from scipy.interpolate import CubicSpline,interp1d
import matplotlib.pyplot as plt

                              
def parse_data(path):
    """Load and parse input."""
    data_input=pathlib.Path(path).read_text().rstrip()
    lines = data_input.split('\n')
    sf = np.array([[float(p) for p in line.split()] for line in lines[1:]])
    return(sf)
  

def get_f_dp(f_p):
    """Derive f" (f1) from f' (f2) """
    pass


# def fftkk(dF_in, E, switch=1, grain=0.05):
#     #Pad out the spectrum by 5000 points
#     k = 5000
#     E_dn = np.arange(E[0] - grain, E[0] - (k+1)*grain, -grain)
#     E_dn = np.fliplr(E_dn.reshape(1, E_dn.shape[0])).reshape(E_dn.shape[0],)
#     E_up = np.arange(E[-1] + grain, E[-1] + (k+1)*grain, grain)
#     _E_ = np.hstack((E_dn, E, E_up))
#     #Take edge of spectrum gently to zero using a quarter of a sine wave
#     range = np.linspace(-np.pi/2, np.pi/2, k)
#     dn = (np.sin(range) / 2) + 0.5
#     up = 1 - dn
#     dF_dn = dF_in[0] * dn
#     dF_up = dF_in[-1] * up
#     _dF_ = np.hstack((dF_dn, dF_in, dF_up))
#     #This is based on fftkk.f by Graham George which
#     #in turn is based on a paper.
#     npts = _E_.shape[0]
#     Hz = np.fft.fft(_dF_)
#     mn = Hz[0]
#     front = Hz[0:(len(Hz)//2)]
#     tmp = front.copy()
#     #I believe the next line is the magic.  Its the convolution with the signum or
#     #how to bypass the difficult integration.
#     tmp.real, tmp.imag = front.imag, front.real
#     front = switch * tmp
#     back = Hz[(len(Hz)//2):]
#     tmp2 = back.copy()
#     tmp2.real, tmp2.imag = back.imag, back.real
#     back = -switch * tmp2
#     new_Hz = np.hstack((mn, front, back))
#     dF = np.fft.ifft(new_Hz).reshape((1, npts))
#     dF_out = np.fliplr(dF).reshape((npts,))
#     return dF_out[k:-k].real

def get_f_p(energy, f_dp, padn=5000):
    """Derive f' (f2) from f" (f1) """
    
    denergy = energy[1:]-energy[:-1]
    if np.any(denergy-denergy[0]): # nonuniform spacing
        uniform_mesh = np.arange(energy[0],energy[-1],energy[1]-energy[0])
        interp = interp1d(energy, f_dp)
        F=interp(uniform_mesh) # interpolated f_dp
        energy=uniform_mesh
    else:
        F = f_dp   
    
    dE = energy[1]-energy[0]
    
    # sin padding as used in Sherrell thesis
    S = np.sin(np.linspace(-np.pi / 2, np.pi / 2, padn)) * 0.5 + 0.5
    Fin = np.hstack((F[0] * np.ones_like(S),
                      F,
                      F[-1] * np.ones_like((1 - S))))  

    if padn==0:
        f_p_pred = np.imag(dE*hilbert(Fin, N=None, axis=- 1))
    else:
        f_p_pred = np.imag(dE*hilbert(Fin, N=None, axis=- 1))[padn:-padn]
    
    return(energy,f_p_pred)

def plot_f_p(energy, f_p, f_dp, padn=5000):
    """Derive f' (f2) from f" (f1) and plot comparison"""
    
    energy_interp,f_p_pred = get_f_p(energy, f_dp, padn=padn)
    
    # interpolation in case of nonuniform energy
    f_p_interp = interp1d(energy, f_p)(energy_interp)

    plt.figure()
    plt.plot(energy_interp, f_p_interp)
    plt.plot(energy_interp, f_p_pred)
    
    return(energy_interp,f_p_interp, f_p_pred)
    

if __name__ == "__main__":
    path = 'sample_data/pf-rd-ox_fftkk.out'
    sf = parse_data(path)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]
    
    start_ind = 6063
    stop_ind = 6163
    # 7070–7170 eV range
    energy = energy[start_ind:stop_ind]
    f_p = f_p[start_ind:stop_ind]
    f_dp = f_dp[start_ind:stop_ind]
    
    energy_interp,f_p_interp, f_p_pred = plot_f_p(energy, f_p, f_dp, padn=5000)

    # f_p2 = fftkk(f_dp, energy, switch=1, grain=0.05)
    # plt.figure()
    # plt.plot(f_p_interp)
    # plt.plot(f_p2)
    
    # breakpoint()
    # start_ind = 6063
    # stop_ind = 6163
    
    # plt.figure()
    # plt.plot(energy_interp[start_ind:stop_ind], f_p_interp[start_ind:stop_ind])
    # plt.plot(energy_interp[start_ind:stop_ind], f_p_pred[start_ind:stop_ind])
    
    path = 'sample_data/pf-rd-red_fftkk.out'
    sf = parse_data(path)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]
    
    # 7070–7170 eV range
    energy = energy[start_ind:stop_ind]
    f_p = f_p[start_ind:stop_ind]
    f_dp = f_dp[start_ind:stop_ind]
    
    energy_interp,f_p_interp, f_p_pred = plot_f_p(energy, f_p, f_dp, padn=5000)
    

    # f_dp = spectrum-F2

#     for path in sys.argv[1:]:
#         print(f"\n{path}:")
#         solutions = solve(puzzle_input=pathlib.Path(path).read_text().rstrip())
#         print("\n".join(str(solution) for solution in solutions))
