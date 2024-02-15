import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits
from cobaya.model import get_model
from cobaya.run import run

l_max = 1300
nside = 512
f_sky = 0.85

#cmb plus noise
Cls = np.asarray(pd.read_csv("../../PolSpice/PolSpice_v03-07-05/cl_hp_apodize.dat", skiprows = [0], 
                             delim_whitespace=True, 
                             names = ["TT", "EE", "BB", "TE", "TB", "EB", "ET", "BT", "BE"]).iloc[:, 0:6])
Cls_map_lowl = Cls[2:30].flatten('F')

#just noise
noise = fits.open('/mn/stornext/d16/cmbco/bp/johanres/commander_camb/camb_data/tod_070_rms_c0001_k000797_nside512.fits')
noise_data = noise[1].data
noise_temp = noise_data['TEMPERATURE'].flatten()
noise_q    = noise_data['Q_POLARISATION'].flatten()
noise_u    = noise_data['U_POLARISATION'].flatten()

sigma_T = np.sqrt(np.mean(noise_temp**2))
sigma_P = np.sqrt(np.mean(noise_q**2))

N_l_T = sigma_T**2 * 4.*np.pi/(12.*nside**2)
N_l_P = sigma_P**2 * 4.*np.pi/(12.*nside**2)

#beam data and pixel window
beam = fits.open('Bl_TEB_npipe6v19_70GHzx70GHz.fits')[1]
pixel_window_T, pixel_window_P = np.array(hp.sphtfunc.pixwin(nside = nside, pol = True, lmax = l_max))

func_tt = (beam.data['T'][:l_max+1] * pixel_window_T)**2
func_ee = (beam.data['E'][:l_max+1] * pixel_window_P)**2
func_bb = (beam.data['B'][:l_max+1] * pixel_window_P)**2
func_te = pixel_window_T * pixel_window_P * beam.data['T'][:l_max+1] * beam.data['E'][:l_max+1]
func_eb = pixel_window_P**2 * beam.data['E'][:l_max+1] * beam.data['B'][:l_max+1]
func_tb = pixel_window_T * pixel_window_P * beam.data['T'][:l_max+1] * beam.data['B'][:l_max+1]

#PolSpice
M_ell_all = np.loadtxt(f"../../PolSpice/PolSpice_v03-07-05/M_ell_gal.txt")

#get each sub-matrix
M_ell_all = M_ell_all.reshape(6, 46, -1, 46).swapaxes(1, 2).reshape(-1, 46, 46)

#crop to ell<30
M_ell = np.zeros((36, 28, 28))
M_ell[:, :, :] = M_ell_all[:, :28, :28]

#set off-diagonal elements to 0
diag = np.arange(6)*7
for i in range(36):
    if i in diag:
        M_ell[i, :, :] = M_ell[i, :28, :28]
        
#shape back into normal
M_ell = M_ell.reshape(6, -1, 28, 28).swapaxes(1, 2).reshape(168, 168)

#likelihood function
def all_like(_self=None):
    ells = np.arange(l_max+1)
    
    Cls_camb = _self.provider.get_Cl(ell_factor=False, units = "muK2")
    Cls_camb_tt = np.asarray(Cls_camb['tt'][:l_max+1]*func_tt + N_l_T)
    Cls_camb_ee = np.asarray(Cls_camb['ee'][:l_max+1]*func_ee + N_l_P)
    Cls_camb_bb = np.asarray(Cls_camb['bb'][:l_max+1]*func_bb + N_l_P)
    Cls_camb_te = np.asarray(Cls_camb['te'][:l_max+1]*func_te)
    
    
    Cls_map_theo = np.concatenate((np.stack((Cls_camb_tt, Cls_camb_ee, Cls_camb_bb, Cls_camb_te), axis = 1), 
                                   np.zeros((l_max+1, 2))), axis = 1)
    Cls_theo_lowl = Cls_map_theo[2:30].flatten('F')
    """
    Cls_map_highl = np.zeros((l_max+1, 6))
    Cls_theo_highl = np.zeros((l_max+1, 6))
    for ell in range(l_max+1):
        Cls_map_highl[ell] = np.array([Cls[ell, 0], Cls[ell, 1], Cls[ell, 2], Cls[ell, 3], Cls[ell, 4], Cls[ell, 5]])
        Cls_theo_highl[ell] = np.array([Cls_camb_tt[ell], Cls_camb_ee[ell], Cls_camb_bb[ell],   Cls_camb_te[ell], 0., 0.])
    """
    
    Cls_map = np.zeros((l_max+1, 3, 3))
    Cls_map_theo = np.zeros((l_max+1, 3, 3))
    for ell in range(l_max+1):
        Cls_map[ell] = np.array([[Cls[ell, 0], Cls[ell, 3], Cls[ell, 4]],
                                 [Cls[ell, 3], Cls[ell, 1], Cls[ell, 5]],
                                 [Cls[ell, 4], Cls[ell, 5], Cls[ell, 2]]])
        Cls_map_theo[ell] = np.array([[Cls_camb_tt[ell], Cls_camb_te[ell], 0.              ],
                                      [Cls_camb_te[ell], Cls_camb_ee[ell], 0.              ],
                                      [0.,               0.,               Cls_camb_bb[ell]]])

    #Compute the binned log-likelihood:
    #low ell first:
    C_ell = Cls_map_lowl - Cls_theo_lowl
    logp = -np.dot(np.dot(C_ell, np.linalg.inv(M_ell)), C_ell)/2.
    
    #high ell: Gaussian approximation for correlated fields
    """
    for ell in range(30, l_max+1):
        const = -(2*ell + 1) * f_sky**2
        logp += const * (((Cls_map_highl[ell, 0] - Cls_theo_highl[ell, 0])/Cls_theo_highl[ell, 0])**2
                         + ((Cls_map_highl[ell, 1] - Cls_theo_highl[ell, 1])/Cls_theo_highl[ell, 1])**2
                         + ((Cls_map_highl[ell, 2] - Cls_theo_highl[ell, 2])/Cls_theo_highl[ell, 2])**2
                         + ((Cls_map_highl[ell, 3] - Cls_theo_highl[ell, 3])/np.sqrt(Cls_theo_highl[ell, 0]*Cls_theo_highl[ell, 1]))**2
                         + ((Cls_map_highl[ell, 4] - Cls_theo_highl[ell, 4])/np.sqrt(Cls_theo_highl[ell, 0]*Cls_theo_highl[ell, 2]))**2
                         + ((Cls_map_highl[ell, 5] - Cls_theo_highl[ell, 5])/np.sqrt(Cls_theo_highl[ell, 1]*Cls_theo_highl[ell, 2]))**2)
    """
    
    #OR we try wishart distribution like before for high ell
    for ell in range(30, l_max+1):
        lndet1 = (2*ell-3)/2*np.linalg.slogdet(Cls_map[ell])[1]
        lndet2 = (2*ell+1)/2*np.linalg.slogdet(Cls_map_theo[ell])[1]
        Tr = (2*ell+1)/2*np.trace(Cls_map[ell].dot(np.linalg.inv(Cls_map_theo[ell])))
        logp += lndet1 - lndet2 - Tr
    return logp

