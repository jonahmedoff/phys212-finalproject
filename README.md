Jonah Medoff
# phys212-finalproject
Code used for Cosmology Final Project:
MCMC Likelihood Analysis of Planck 2018 Data

Planck_MCMC.ipynb is a juptyter notebook containing all of the code used in this project
(e.g., downloading Planck data, MCMC algorithm, generating plots, etc.).
Aside from downloading the Planck data (which can be done using the Planck Legacy Archive), it should be completely runnable.

planck_mcmc.py is a python script version of the MCMC algorithm, which can be used to run multiple MCMC analyses at the same time.
The inputs are the 6 starting parameters, number of iterations, and ID number (i.e., a number that will appear in the output filename).

The remaining files are the chains outputted from my runs of the MCMC code.
