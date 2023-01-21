"""Plots to illustrate Kramers Kronig relations"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import core_functions

"""Set text sizes"""
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_inds(energy, start_eV, end_eV):
    start_ind = np.argmin(np.abs(energy-start_eV))
    end_ind = np.argmin(np.abs(energy-end_eV))
    return(start_ind,end_ind)


if __name__ == "__main__":
    
    """Henke data from https://henke.lbl.gov/optical_constants/"""

    path='sample_data/Fe.nff' # Henke data
    sf = core_functions.parse_data(path, remove_first_line=True)
    energy = sf[:,0]
    f_p = sf[:,1]
    f_dp = sf[:,2]

    energy_interp,f_p_pred = core_functions.get_f_p(energy, f_dp, padn=5000,
                                                    Z = 26, # atomic number
                                                    include_Z_term=True,
                                                    window_type='cosine',
                                                    )

    start_eV = 5000
    end_eV = 25000
    
    start_ind,end_ind = get_inds(energy, start_eV, end_eV)
    start_ind_interp,end_ind_interp = get_inds(energy_interp, energy[start_ind], energy[end_ind])
    
    plt.figure()
    plt.plot(energy[start_ind:end_ind],f_dp[start_ind:end_ind],linewidth=2,label='Table Value')
    plt.savefig('plots/henke_f_dp.png',dpi=300, pad_inches=0.0)
    
    plt.figure()
    plt.plot(energy[start_ind:end_ind],f_p[start_ind:end_ind],linewidth=2,label='Table Value')
    plt.plot(energy_interp[start_ind_interp:end_ind_interp],f_p_pred[start_ind_interp:end_ind_interp],'-',linewidth=2, label='Computed from Kramers Kronig')
    plt.legend()
    plt.savefig('plots/henke_f_p.png',dpi=300, pad_inches=0.0)
    
    
    
    """Reproduction of Fig 1 in Sauter (2020), https://doi.org/10.1107/S2059798320000418"""
    
    start_eV = 7070
    end_eV = 7170
    
    path='sample_data/Fe_fake.dat' # Henke data
    sf_fe_0 = core_functions.parse_data(path, remove_first_line=False)
    start_ind_fe_0, end_ind_fe_0 = get_inds(sf_fe_0[:,0], start_eV, end_eV)

    path='sample_data/pf-rd-red_fftkk.out'
    sf_fe_2 = core_functions.parse_data(path)
    start_ind_fe_2, end_ind_fe_2 = get_inds(sf_fe_2[:,0], start_eV, end_eV)

    path = 'sample_data/pf-rd-ox_fftkk.out'
    sf_fe_3 = core_functions.parse_data(path)
    start_ind_fe_3, end_ind_fe_3 = get_inds(sf_fe_3[:,0], start_eV, end_eV)


    plt.figure()
    plt.plot(sf_fe_0[:,0][start_ind_fe_0:end_ind_fe_0],sf_fe_0[:,2][start_ind_fe_0:end_ind_fe_0],'m', label="Fe0")
    plt.plot(sf_fe_2[:,0][start_ind_fe_2:end_ind_fe_2],sf_fe_2[:,2][start_ind_fe_2:end_ind_fe_2],'r', label="Fe2+")
    plt.plot(sf_fe_3[:,0][start_ind_fe_3:end_ind_fe_3],sf_fe_3[:,2][start_ind_fe_3:end_ind_fe_3],'b', label="Fe3+")
    plt.legend()
    plt.savefig('plots/sherrell_f_dp.png',dpi=300, pad_inches=0.0)
    
    plt.figure()
    plt.plot(sf_fe_0[:,0][start_ind_fe_0:end_ind_fe_0],sf_fe_0[:,1][start_ind_fe_0:end_ind_fe_0],'m', label="Fe0")
    plt.plot(sf_fe_2[:,0][start_ind_fe_2:end_ind_fe_2],sf_fe_2[:,1][start_ind_fe_2:end_ind_fe_2],'r', label="Fe2+")
    plt.plot(sf_fe_3[:,0][start_ind_fe_3:end_ind_fe_3],sf_fe_3[:,1][start_ind_fe_3:end_ind_fe_3],'b', label="Fe3+")
    plt.ylim([-8.5, -2.5])
    plt.legend()
    plt.savefig('plots/sherrell_f_p.png',dpi=300, pad_inches=0.0)
    
    
    """Previous plot overlaid with calculated f_p from Kramers Kronig"""

    padn = 5000
    window_type = 'cosine'
    
    energy_interp_0,f_p_pred_0 = core_functions.get_f_p(sf_fe_0[:,0],sf_fe_0[:,2], padn=padn,
                                                        Z = 26, # atomic number
                                                        include_Z_term=False,
                                                        window_type=window_type,
                                                        )
    start_ind_fe_interp_0, end_ind_fe_interp_0 = get_inds(energy_interp_0, start_eV, end_eV)

    energy_interp_2,f_p_pred_2 = core_functions.get_f_p(sf_fe_2[:,0],sf_fe_2[:,2], padn=padn,
                                                        Z = 26, # atomic number
                                                        include_Z_term=False,
                                                        window_type=window_type,
                                                        )
    start_ind_fe_interp_2, end_ind_fe_interp_2 = get_inds(energy_interp_2, start_eV, end_eV)
    
    energy_interp_3,f_p_pred_3 = core_functions.get_f_p(sf_fe_3[:,0],sf_fe_3[:,2], padn=padn,
                                                        Z = 26, # atomic number
                                                        include_Z_term=False,
                                                        window_type=window_type,
                                                        )
    start_ind_fe_interp_3, end_ind_fe_interp_3 = get_inds(energy_interp_3, start_eV, end_eV)
    
    plt.figure()
    plt.plot(sf_fe_0[:,0][start_ind_fe_0:end_ind_fe_0],sf_fe_0[:,1][start_ind_fe_0:end_ind_fe_0],'m', label="Fe0")
    plt.plot(sf_fe_2[:,0][start_ind_fe_2:end_ind_fe_2],sf_fe_2[:,1][start_ind_fe_2:end_ind_fe_2],'r', label="Fe2+")
    plt.plot(sf_fe_3[:,0][start_ind_fe_3:end_ind_fe_3],sf_fe_3[:,1][start_ind_fe_3:end_ind_fe_3],'b', label="Fe3+")
    
    plt.plot(energy_interp_0[start_ind_fe_interp_0:end_ind_fe_interp_0],f_p_pred_0[start_ind_fe_interp_0:end_ind_fe_interp_0],'m--', label="Fe0 calculated")
    plt.plot(energy_interp_2[start_ind_fe_interp_2:end_ind_fe_interp_2],f_p_pred_2[start_ind_fe_interp_2:end_ind_fe_interp_2],'r--', label="Fe2+ calculated")
    plt.plot(energy_interp_3[start_ind_fe_interp_3:end_ind_fe_interp_3],f_p_pred_3[start_ind_fe_interp_3:end_ind_fe_interp_3],'b--', label="Fe3+ calculated")
    plt.ylim([-8.5, -2.5])
    plt.legend()
    plt.savefig('plots/sherrell_f_p_overlaid.png',dpi=300, pad_inches=0.0)
    
    
    """How does estimate for f_p change with different window LENGTHS for the range 7070-7170eV"""
    
    start_eV = 7070
    end_eV = 7170
    
    energy = sf_fe_2[:,0]
    start_ind, end_ind = get_inds(energy, start_eV, end_eV)
    energy = energy[start_ind:end_ind]
    f_p = sf_fe_2[:,1][start_ind:end_ind]
    f_dp = sf_fe_2[:,2][start_ind:end_ind]
    
    padn_vec = np.arange(0,5)
    padn_vec = 10**padn_vec
    
    plt.figure()
    
    for padn in padn_vec:
        energy_interp,f_p_pred = core_functions.get_f_p(energy,f_dp, padn=padn,
                                                            Z = 26, # atomic number
                                                            include_Z_term=False,
                                                            window_type='cosine',
                                                            )
        plt.plot(energy_interp,f_p_pred,label=str(padn))
        
    plt.legend()
    plt.savefig('plots/f_p_window_lengths.png',dpi=300, pad_inches=0.0)


    """How does estimate for f_p change with different window TYPES for the range 7070-7170eV"""

    window_vec = ['boxcar', 'triang', 'blackman', 'hamming', 'cosine', 'tukey']


    
    plt.figure()
    
    for window_type in window_vec:
        energy_interp,f_p_pred = core_functions.get_f_p(energy,f_dp, padn=5000,
                                                            Z = 26, # atomic number
                                                            include_Z_term=False,
                                                            window_type=window_type,
                                                            )
        plt.plot(energy_interp,f_p_pred,label=window_type)
        
    plt.legend()
    plt.savefig('plots/f_p_window_types.png',dpi=300, pad_inches=0.0)