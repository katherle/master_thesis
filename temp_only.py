import numpy as np
import healpy as hp
from astropy.io import fits
from cobaya.model import get_model
from cobaya.run import run
from cobaya.model import get_model

l_max = 1600 


#cmb plus noise
cmbsim = fits.open('70_noise_cmb_2rms.fits')
cmbsim = cmbsim[1].data
cmbtemp = cmbsim['TEMPERATURE'].flatten()
Cl_tt = hp.sphtfunc.anafast(cmbtemp, lmax = l_max)

#noise only
noise = fits.open('tod_070_rms_c0001_k000797_2rms.fits')
noise_data = noise[1].data
noise_temp = noise_data['TEMPERATURE'].flatten()
noise_temp = noise_temp * np.random.normal(0, 1, len(noise_temp))
Nl_tt = hp.sphtfunc.anafast(noise_temp, lmax = l_max)

#beam data and pixel window
beam = fits.open('Bl_TEB_npipe6v19_70GHzx70GHz.fits')[1]
pixel_window = np.array(hp.pixwin(nside = 1024, pol = False, lmax = l_max))
func_tt = (beam.data['T'][:1601] * pixel_window)**2

def TT_like(_self=None):
    ells = np.arange(l_max+1)
    Cl_map = Cl_tt #with noise and beam
    Nl = Nl_tt
    
    Cl_camb = _self.provider.get_Cl(ell_factor=False, units="muK2")['tt'][:l_max+1]
    
    Cl_est = Cl_camb*func_tt
    Cl_map_est = Cl_est + Nl #with noise, pixel window, and beam
    
    #Compute the log-likelihood
    V = Cl_map[2:]/Cl_map_est[2:]
    logp = np.sum((2*ells[2:] + 1) * (-V/2 + 1/2.*np.log(V)))
    
    return logp