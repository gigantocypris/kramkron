"""Penalty for f' and f" violating Kramers Kronig relations, written in PyTorch"""

import numpy as np
import torch

from core_functions import INTERP_FUNC
from scipy.signal.windows import get_window

def hilbert(x, 
            axis=-1):

    N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = torch.fft.fft(x, N, axis=axis)
    h = torch.zeros(N, dtype=Xf.dtype, requires_grad=False)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = torch.fft.ifft(Xf * h, axis=axis)
    return(x.imag)

def get_f_p(energy, # uniform spacing
            f_dp, 
            padn=5000,
            trim=0,
            window_type='cosine',
            known_response_energy=None,
            known_response_f_p=None,
            known_response_f_dp=None,
            ):
    """Derive f' from f" """
    
    if known_response_energy is not None:
        known_response_f_p_interp = INTERP_FUNC(known_response_energy, known_response_f_p)(energy)
        known_response_f_dp_interp = INTERP_FUNC(known_response_energy, known_response_f_dp)(energy)
    else:
        known_response_f_p_interp = torch.zeros_like(f_dp)
        known_response_f_dp_interp = torch.zeros_like(f_dp)
    
    f_in = f_dp - known_response_f_dp_interp
        
    f_in = apply_window(f_dp, padn, trim=trim, window_type=window_type)
    
    f_p_pred_padded = hilbert(f_in)
    
    if padn != 0:
        f_p_pred = f_p_pred_padded[padn:-padn]
    else:
        f_p_pred = f_p_pred_padded
        
    f_p_pred = f_p_pred + known_response_f_p_interp
    return(f_p_pred,f_p_pred_padded)


def get_f_dp(energy, # uniform spacing
             f_p,
             padn=5000,
             trim=0,
             window_type='cosine',
             known_response_energy=None,
             known_response_f_p=None,
             known_response_f_dp=None,
             ):
    
    """Derive f" from f' """
    
    if known_response_f_p is not None:
        known_response_f_p = -known_response_f_p
        
    f_dp_pred,f_dp_pred_padded = get_f_p(energy, -f_p, padn=padn,
                                         trim=trim,
                                         window_type=window_type,
                                         known_response_energy=known_response_energy,
                                         known_response_f_p=known_response_f_dp,
                                         known_response_f_dp=known_response_f_p,
                                         )
    return(f_dp_pred,f_dp_pred_padded)


def apply_window(f_in, padn,
                 trim=0,
                 window_type='cosine'):
    """
    Possible window_type:
    boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman,
    blackman, harris, nuttall, barthann, cosine, exponential, tukey, taylor,
    lanczos
    """

    window = get_window(window_type,(padn+trim)*2)
    window = window[0:padn+trim]
    window = torch.tensor(np.expand_dims(window, 0))

    
    window_start = f_in[0]*torch.ones(padn+trim)
    window_start[padn:]=f_in[0:trim]
    
    window_end = f_in[-1]*torch.ones(padn+trim)
    window_end[0:trim]=f_in[len(f_in)-trim:]
    
    windowed_func = torch.hstack((window_start * torch.squeeze(window),
                                  f_in[trim:len(f_in)-trim],
                                  window_end * torch.squeeze(torch.fliplr(window))))

    return(windowed_func) 
    
    
    
def penalty(energy, f_p, f_dp, trim=0, padn=5000,window_type='cosine',
            known_response_energy=None,
            known_response_f_p=None,
            known_response_f_dp=None,
            ):
    """How close f' and f" are to obeying the Kramers Kronig relation?"""
    
    """Going from f_dp to f_p"""
    f_p_pred,f_p_pred_padded = get_f_p(energy, f_dp, trim=trim, padn=padn,
                                       window_type=window_type,
                                       known_response_energy=known_response_energy,
                                       known_response_f_p=known_response_f_p,
                                       known_response_f_dp=known_response_f_dp,
                                       )
    
    # add back DC term
    F_p_pred = torch.fft.fft(f_p_pred_padded)
    F_p_pred[0] = torch.fft.fft(f_p)[0]
    f_p_pred_padded = torch.fft.ifft(F_p_pred).real
    
    f_p_pred_padded = f_p_pred_padded[padn:len(f_p_pred_padded)-padn]
    f_p_pred = f_p_pred_padded[trim:len(f_p_pred_padded)-trim]
    
    # trim f_p
    f_p = f_p[trim:len(energy)-trim]
    

    
    """Going from f_p to f_dp"""
    f_dp_pred,f_dp_pred_padded = get_f_dp(energy, f_p, trim=trim, padn=padn,
                                          window_type=window_type,
                                          known_response_energy=known_response_energy,
                                          known_response_f_p=known_response_f_p,
                                          known_response_f_dp=known_response_f_dp,
                                          )
    
    # add back DC term
    F_dp_pred = torch.fft.fft(f_dp_pred_padded)
    F_dp_pred[0] = torch.fft.fft(f_dp)[0]
    f_dp_pred_padded = torch.fft.ifft(F_dp_pred).real
    
    f_dp_pred_padded = f_dp_pred_padded[padn:len(f_dp_pred_padded)-padn]
    f_dp_pred = f_dp_pred_padded[trim:len(f_dp_pred_padded)-trim]
    
    # trim f_dp
    f_dp = f_dp[trim:len(energy)-trim]

    mse = torch.mean((f_p - f_p_pred)**2) + torch.mean((f_dp - f_dp_pred)**2)
    return(mse)
