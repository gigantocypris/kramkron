"""
A great approach would be to work out a unit test first, then actually implement the code.
The test would simulate f″ based on a very simple model of the K-edge.
Then use the dispersion relations to calculate f′.
Then sample both of these curves with Gaussian noise to simulate experimental measurement of the two curves.
Then develop a restraint model, and optimize the parameters. Presumably use automatic differentiation for first-derivatives.
Compare the optimized model to the initial ground truth (and pass the test based on a tolerance). Show result in matplotlib.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import core_functions
import core_functions_pytorch

def create_f(width=10,
             dE=.1,
             trim=0,
             slope = 1,
             padn=5000,
             ):

    energy = np.arange(-width,width,dE)

    """ramp/unit step function for f" to emulate a simple K-edge"""
    ramp = energy*slope
    ramp_start = np.argmin(np.abs(ramp))
    ramp_end = np.argmin(np.abs(ramp-1))
    ramp_size = ramp_end-ramp_start
    f_dp = np.heaviside(energy, 0.5)
    mid_ind = len(f_dp)//2
    f_dp[mid_ind:mid_ind+ramp_size] = ramp[ramp_start:ramp_end]
    
    """get f' from the Hilbert transform"""
    _,f_p,_,_,_ = \
    core_functions.get_f_p(energy, f_dp, padn=padn,
                           trim=trim,
                           )
    
    energy = energy[trim:len(energy)-trim]
    f_dp = f_dp[trim:len(f_dp)-trim]
    f_p = f_p[trim:len(f_p)-trim]
    
    return(energy,f_p,f_dp)

def sample(f_p,
           f_dp,
           loc=[0,0],
           scale=[1e-2,1e-2],
           ):
    f_p_dist = torch.distributions.normal.Normal(f_p + loc[0], scale[0])
    f_dp_dist = torch.distributions.normal.Normal(f_dp + loc[1], scale[1])
    return(f_p_dist.sample(),f_dp_dist.sample())


def subsample(energy,
              f_p,
              f_dp,
              spacing=2):
    inds = np.arange(0,len(energy),spacing)
    energy = energy[inds]
    f_p = f_p[inds]
    f_dp = f_dp[inds]
    return(energy,f_p,f_dp,inds)

def loss_fn(f_p_opt,
            f_dp_opt,
            f_p_noisy_ss,
            f_dp_noisy_ss,
            inds,
            padn):
    data_loss = torch.mean((f_p_opt[inds]-f_p_noisy_ss)**2 + (f_dp_opt[inds]-f_dp_noisy_ss)**2)
    kk_loss = core_functions_pytorch.penalty(energy, f_p_opt, f_dp_opt, padn=padn)
    
    return(data_loss + kk_loss)


if __name__ == "__main__":

    width=5
    padn=100
    trim=30
    energy,f_p,f_dp = create_f(width=width,
                               padn=padn,
                               trim=trim)

    f_p = torch.tensor(f_p)
    f_dp = torch.tensor(f_dp)
    
    f_p_noisy,f_dp_noisy = sample(f_p,f_dp)
    
    energy_ss,f_p_noisy_ss,f_dp_noisy_ss,inds = subsample(energy,f_p_noisy,f_dp_noisy,
                                                          spacing=1)
    
    plt.figure()
    plt.plot(energy,f_dp,energy_ss,f_dp_noisy_ss,'.')
    plt.plot(energy,f_p,energy_ss,f_p_noisy_ss,'.')
    plt.xlim([-width,width])
    
    core_functions_pytorch.penalty(energy, f_p, f_dp, padn=padn)

    """From energy_ss,f_p_noisy_ss,f_dp_noisy_ss determine f_p and f_dp, energy is given"""
    
    f_p_pred_0 = core_functions.INTERP_FUNC(energy_ss,f_p_noisy_ss)(energy)
    f_dp_pred_0 = core_functions.INTERP_FUNC(energy_ss,f_dp_noisy_ss)(energy)
    
    f_p_opt = torch.tensor(f_p_pred_0,requires_grad=True)
    f_dp_opt = torch.tensor(f_dp_pred_0, requires_grad=True)
    
    plt.figure()
    plt.plot(energy,f_dp,energy,f_dp_opt.detach().numpy())
    plt.plot(energy,f_p,energy,f_p_opt.detach().numpy())
    plt.xlim([-width,width])
    
    learning_rate = 1e-2
    num_iter = 1000

    
    optimizer = torch.optim.SGD([f_p_opt,f_dp_opt],lr=learning_rate)
    
    for i in range(num_iter):
        loss = loss_fn(f_p_opt,
                       f_dp_opt,
                       f_p_noisy_ss,
                       f_dp_noisy_ss,
                       inds, padn)
        if i%100:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    plt.figure()
    plt.plot(energy,f_dp,energy,f_dp_opt.detach().numpy())
    plt.plot(energy,f_p,energy,f_p_opt.detach().numpy())
    plt.xlim([-width,width])
    
    
    