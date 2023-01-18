#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:47:31 2023

@author: vganapa1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:32:57 2022

@author: vganapa1

References for Kramers-Kronig equation:
  https://aip.scitation.org/doi/10.1063/1.4747813
  http://skuld.bmsc.washington.edu/scatter/AS_kk.html
  real data at: http://skuld.bmsc.washington.edu/scatter/data/Fe.dat
  
"""


from LS49_pytorch.sim.fdp_plot import george_sherrell
import os
import libtbx.load_env


Fe_oxidized_model = george_sherrell(os.path.join(libtbx.env.find_in_repositories("ls49_big_data"),"data_sherrell/pf-rd-ox_fftkk.out")),
Fe_reduced_model = george_sherrell(os.path.join(libtbx.env.find_in_repositories("ls49_big_data"),"data_sherrell/pf-rd-red_fftkk.out")),
Fe_metallic_model = george_sherrell(os.path.join(libtbx.env.find_in_repositories("ls49_big_data"),"data_sherrell/Fe_fake.dat")),


breakpoint()
import numpy as np
from scipy.signal import convolve, hilbert
import matplotlib.pyplot as plt

energy_lb = 0 # eV
energy_ub = 7170+2000 # eV
d_eV = 1
abs_edge = 7110 # absorption edge
smoothing_kernel = np.ones([5])
smoothing_kernel = smoothing_kernel/np.sum(smoothing_kernel)
start_plot = 7070 - energy_lb
stop_plot = 7170 - energy_lb
breakpoint()
# energy vector
energy_vec = np.arange(energy_lb, energy_ub, d_eV)

# step function
# apply smoothing function to step function

f_dp = np.zeros_like(energy_vec)
f_dp[energy_vec < abs_edge] = 0.5
f_dp[energy_vec >= abs_edge] = 4.0

# final f"
f_dp = convolve(f_dp, smoothing_kernel, mode='valid', method='auto')

plt.figure()
plt.plot(np.arange(start_plot, stop_plot), f_dp[start_plot:stop_plot])


# kramer's kronig to get f'

f_p = hilbert(f_dp, N=None, axis=- 1)

plt.figure()
plt.plot(np.arange(start_plot, stop_plot), np.real(f_p)[start_plot:stop_plot])

plt.figure()
plt.plot(np.arange(start_plot, stop_plot), np.imag(f_p)[start_plot:stop_plot])


### diffBragg
def f_prime(f_double_prime, S=None, padn=5000):
    """
    generate an f_prime from an f_double_prime curve
    using the kramers kronig relationship
    :param f_double_prime: function or its derivative
    :param S:
    :param padn:
    :return:
    """
    if S is None:
        S = np.sin(np.linspace(-np.pi / 2, np.pi / 2, padn)) * 0.5 + 0.5
    else:
        padn = S.shape[0]
    F = f_double_prime
    Fin = np.hstack((F[0] * S,
                     F,
                     F[-1] * (1 - S)))  # sin padding as used in Sherrell thesis, TODO window function ?
    Ft = fft.fft(Fin)
    iFt = -1 * fft.ifft(1j * np.sign(fft.fftfreq(Ft.shape[0])) * Ft).real
    return iFt[padn:-padn]


### henke tables
'''
Access to Henke tables.
  /*! Henke tables are available for elements with Z=1-92.
      Each table contains 500+ points on a uniform logarithmic mesh
      from 10 to 30,000 eV with points added 0.1 eV above and below
      "sharp" absorption edges. The atomic scattering factors are
      based upon experimental measurements of the atomic
      photoabsorption cross section. The absorption measurements
      provide values for the imaginary part of the atomic scattering
      factor. The real part is calculated from the absorption
      measurements using the Kramers-Kronig integral relations.
      <p>
      Reference: B. L. Henke, E. M. Gullikson, and J. C. Davis,
      Atomic Data and Nuclear Data Tables Vol. 54 No. 2 (July 1993).<br>
      ftp://grace.lbl.gov/pub/sf/
      <p>
      See also:
        http://www-cxro.lbl.gov/optical_constants/asf.html <br>
        http://www.esrf.fr/computing/scientific/dabax/
   */
'''  
###

# check kramer's kronig
from scipy.signal import hilbert
fp_oxidized_kk = hilbert(fdp_oxidized, N=None, axis=- 1)

plt.figure()
plt.plot(wavlen2, fp_oxidized)
plt.plot(wavlen2, np.imag(fp_oxidized_kk)) # XXX curves don't quite match
plt.show()

# XXX cannot enforce kramer's kronig over a short segment of the function, but can possibly use kramer's kronig as a soft constraint