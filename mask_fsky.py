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
Cls = pd.read_csv("../../PolSpice/PolSpice_v03-07-05/cl_hp_apodize.dat", skiprows = [0], delim_whitespace=True, 
                  names = ["TT", "EE", "BB", "TE", "TB", "EB", "ET", "BT", "BE"])
Cl_tt = np.array(Cls["TT"])
Cl_ee = np.array(Cls["EE"])
Cl_bb = np.array(Cls["BB"])
Cl_te = np.array(Cls["TE"])
Cl_eb = np.array(Cls["EB"])
Cl_tb = np.array(Cls["TB"])

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

#likelihood function
def all_like(_self=None):
    ells = np.arange(l_max+1)
    
    Cls_camb = _self.provider.get_Cl(ell_factor=False, units = "muK2")
    camb_tt = Cls_camb['tt'][:l_max+1] * func_tt 
    camb_ee = Cls_camb['ee'][:l_max+1] * func_ee 
    camb_bb = Cls_camb['bb'][:l_max+1] * func_bb 
    camb_te = Cls_camb['te'][:l_max+1] * func_te 
    
    Cls_map = np.zeros((l_max+1, 3, 3))
    Cls_map_theo = np.zeros((l_max+1, 3, 3))
    for ell in range(l_max+1):
        Cls_map[ell] = np.array([[Cl_tt[ell], Cl_te[ell], Cl_tb[ell]],
                                 [Cl_te[ell], Cl_ee[ell], Cl_eb[ell]],
                                 [Cl_tb[ell], Cl_eb[ell], Cl_bb[ell]]])
        Cls_map_theo[ell] = np.array([[camb_tt[ell] + N_l_T, camb_te[ell],         0.                  ],
                                      [camb_te[ell],         camb_ee[ell] + N_l_P, 0.                  ],
                                      [0.,                   0.,                   camb_bb[ell] + N_l_P]])
        
    #Compute the log-likelihood from a Wishart distribution:
    logp = 0
    for ell in range(2, l_max+1):
        lndet1 = (2*ell-3)/2*np.linalg.slogdet(Cls_map[ell])[1]
        lndet2 = (2*ell+1)/2*np.linalg.slogdet(Cls_map_theo[ell])[1]
        Tr = (2*ell+1)/2*np.trace(Cls_map[ell].dot(np.linalg.inv(Cls_map_theo[ell])))
        logp += lndet1 - lndet2 - Tr
    
    return logp*f_sky
