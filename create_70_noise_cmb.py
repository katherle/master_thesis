import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import camb

nside=1024
noise_map = hp.read_map('camb_data/tod_070_rms_c0001_k000797.fits', field=(0,1,2))
print(noise_map.shape)
random = np.random.normal(size=noise_map.shape)*noise_map


print(random[:, :5])
print(noise_map[:, :5])
print(random.shape)

beam = fits.open('camb_data/Bl_TEB_npipe6v19_70GHzx70GHz.fits')[1]

l_max=1600
#cp = camb.set_params(tau=0.0544, ns=0.9649, H0=67.36, ombh2=0.02237, omch2=0.12, As=2.1e-09, lmax=l_max)
cp = camb.read_ini('Commander/params.ini')
print(cp)
camb_results = camb.get_results(cp)
all_cls_th = camb_results.get_cmb_power_spectra(lmax=l_max, raw_cl=True, CMB_unit='muK')['lensed_scalar']

alm_cmb = hp.synalm(all_cls_th.transpose(), lmax=l_max, new=True)

c_l_realization = hp.alm2cl(alm_cmb)

def get_D_l(c_l):
        return np.array([c_l[l] * l * (l+1)/(2*np.pi) for l in range(len(c_l))])    
labels_spectra = ['TT', 'EE', 'BB', 'TE']
lmin = 2
i=0
fig, ax = plt.subplots(nrows=2, ncols=2)
for row in ax:
    for col in row:
        col.set_title(labels_spectra[i])
        # TT, EE, BB, TB
        col.plot(get_D_l(all_cls_th[:l_max, i])[lmin:], label='CAMB')
        col.plot(get_D_l(c_l_realization[i, :l_max])[lmin:], label='Realization')
        
        i += 1
plt.legend(bbox_to_anchor =(1.15, 1.5))
plt.savefig('realization_spectra.pdf', bbox_inches='tight')

pixel_window = np.array(hp.pixwin(nside=1024, pol=True, lmax=l_max))
pw2 = fits.open('camb_data/pixel_window_n1024.fits')
beam = fits.open('camb_data/Bl_TEB_npipe6v19_70GHzx70GHz.fits')[1]
for i in tqdm(range(len(alm_cmb[0, :]))):
    l, m = hp.Alm.getlm(lmax=l_max, i=i)
    alm_cmb[0, i] *= beam.data['T'][l] * pixel_window[0, l]
    alm_cmb[1, i] *= beam.data['E'][l] * pixel_window[1, l]
    alm_cmb[2, i] *= beam.data['B'][l] * pixel_window[1, l]

cmb_map = hp.sphtfunc.alm2map(alm_cmb, nside=nside, pol=True)
cmb_map += random

#plt.plot(data_list[0, :])
#cmb_map[0, :] += dipole

hp.mollview(cmb_map[0, :])
plt.savefig('cmb_map.pdf')
hp.mollview(cmb_map[1, :])
hp.mollview(cmb_map[2, :])
hp.write_map('70_noise_cmb.fits', cmb_map, overwrite = True)
#hp.write_map('70__cmb_rms.fits', noise_map, overwrite=True)