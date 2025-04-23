import time
import os
import sys
import numpy as np
import pandas as pd
import healpy as hp
#from tqdm.notebook import tqdm
from astropy.io import fits
from astropy import table
import astropy.units as u
import camb
from scipy.stats import norm

def theory_cl_tt(theta):
    ombh2, omch2, H0, ns, As, tau = theta
    
    params = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,  
                             As=As, ns=ns, halofit_version='mead', lmax=2500)
    results = camb.get_results(params)
    powerspec = results.get_cmb_power_spectra(params, CMB_unit='muK')
    theory_cl_tt = powerspec['total'][:,0]
    
    # This output is dimensionless, so I need to add back the dimension
    ells = np.arange(len(theory_cl_tt))
    theory_cl_tt = np.divide(theory_cl_tt*(2*np.pi), ells*(ells+1), 
                      out=np.zeros_like(theory_cl_tt), where=ells != 0)
    return theory_cl_tt

def log_prior(theta, tau_correction = True):
    ombh2, omch2, H0, ns, As, tau = theta
    
    # Play around with these bounds
    if not (0.01 < ombh2 < 0.04): return -np.inf
    if not (0.05 < omch2 < 0.2): return -np.inf
    if not (50 < H0 < 190): return -np.inf
    if not (0.01 < tau < 0.15): return -np.inf
    if not (0.9 < ns < 1.1): return -np.inf
    if not (1e-9 < As < 3e-9): return -np.inf
    
    if tau_correction:
        # These are the values determined by the Planck low-l E-mode polarization data
        tau_mean = 0.054
        tau_sigma = 0.007
        log_prior_tau = norm.logpdf(tau, loc=tau_mean, scale=tau_sigma)
        return log_prior_tau

    return 0.0  # log(1) for uniform priors

# Gaussian likelihood
# Later I can try using the Planck likelihoods (or just another generic likelihood function)

def log_likelihood(theta, ells, cl_tt_obs, sigma_l):
    ombh2, omch2, H0, ns, As, tau = theta
    
    cl_tt_theory = theory_cl_tt([ombh2, omch2, H0, ns, As, tau])
    cl_tt_theory_interp = np.interp(ells, np.arange(len(cl_tt_theory)), cl_tt_theory)
    
    chi2 = np.sum(((cl_tt_obs - cl_tt_theory_interp)/sigma_l)**2 + np.log(2*np.pi*sigma_l**2))

    return -0.5 * chi2

# posterior = prior*likelihood
# log(posterior) = log(prior) + log(likelihood)

def log_posterior(theta, ells, cl_tt_obs, sigma_l, tau_correction = True):
    lp = log_prior(theta, tau_correction = tau_correction)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta, ells, cl_tt_obs, sigma_l)

def main(argv):
    # main(theta_current, ells, cl_tt_obs, sigma_l, Nsteps=1000, gaussian_std=[1,1,1,1,1,1]):
    
    maps = hp.read_map("COM_CMB_IQU-smica_2048_R3.00_full.fits", field=(0, 5))
    temp_map = maps[0] 
    alm = hp.map2alm(temp_map*1e6, lmax=2500) 
    cl_tt = hp.alm2cl(alm)
    ells = np.arange(len(cl_tt))
    #dl_tt = ells*(ells + 1)/(2*np.pi)*cl_tt
    
    #sigma_l = 0.05 * cl_tt
    #sigma_l[0]=sigma_l[1]
    sigma_l = np.sqrt(2/(2*ells+1)) * cl_tt
    sigma_l[0]=sigma_l[1]
    
    # params: 0.0224 0.12 67.4 0.965 2.1 0.054 4000 0.000224 0.0012 0.674 0.00965 0.021 0.00054 1
    # params: 0.0224 0.12 67.4 0.965 2.1 0.054 4000 1
    #print(sys.argv[1], type(sys.argv[1]), type(float(sys.argv[1])))
    theta_current = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), 
                     float(sys.argv[5])*1e-9, float(sys.argv[6])]
    Nsteps = int(sys.argv[7])
    #gaussian_std = [float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]), float(sys.argv[11]), 
    #                float(sys.argv[12])*1e-9, float(sys.argv[13])]
    gaussian_std = [i*0.01 for i in theta_current]
    #chain_id = str(sys.argv[14])
    chain_id = str(sys.argv[8])
    
    """
    Run Metropolis-Hastings MCMC for Planck parameter estimation
    
    Parameters:
        theta_current: Initial parameter values (array-like)
        ells: Multipole moments
        cl_tt_obs: Observed C_ell^TT
        sigma_l: Uncertainty per ell
        Nsteps: Number of MCMC steps
        gaussian_std: Proposal standard deviation (scalar or array-like)
    
    Returns:
        chain: Array of shape (n_params, Nsteps)
    """
    chain = np.zeros((len(theta_current), Nsteps))
    log_post_current = log_posterior(theta_current, ells, cl_tt, sigma_l, tau_correction=True)
    chain[:,0] = theta_current
    
    print(log_post_current); print()
    accept_count = 0
    count = 0
    
    for step in range(1, Nsteps):
        # Propose new parameters and calculate corresponding acceptance ratio
        theta_new = theta_current + np.random.normal(0,gaussian_std)
        log_post_new = log_posterior(theta_new, ells, cl_tt, sigma_l, tau_correction=True)
        # Metropolis-Hastings uses an acceptance ratio of the form
        # accept_ratio = min(1, posterior(theta_new)/posterior(theta_current))
        log_accept_ratio = log_post_new - log_post_current
        if count % 200 == 0:
            print(log_post_new)
        count += 1
        
        # Accept proposed theta
        if np.log(np.random.uniform(0,1)) < log_accept_ratio:
            theta_current = theta_new
            log_post_current = log_post_new
            accept_count += 1
        
        chain[:,step] = theta_current
    
    print("Acceptance rate:", accept_count / Nsteps)
    print('Done!')
    #return chain
    np.save(f'planck_mcmc_chain_{chain_id}.npy', chain)

if __name__ == "__main__":
    main(sys.argv)