Contains code I wrote for CMB power spectrum analysis using Cobaya as part of my master's thesis. Does not contain the fits files used to run the script; also does not contain a cobaya installation. 

- `setup.ipynb` is the process of initially figuring out how Cobaya works
- `results.ipynb` contains code that compares the final parameter estimates for a variety of different Cobaya runs, as well as Johannes's Commander analysis.
- `Cls_analysis.ipynb` primarily contains the process of fine-tuning the low-ell power spectrum fit using PolSpice.
- all `.py` files are setup for Cobaya runs with various simulated CMBs using the corresponding `.yaml` files: one temperature-only full sky run with isotropic noise, one including polarization, one introducing anisotropic noise and a constant-latitude mask, and one with a mask shaped to block out the brightest regions of the Milky Way specifically.
