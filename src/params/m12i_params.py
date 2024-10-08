# Licensed under BSD-3-Clause License - see LICENSE

import os

import numpy as np

# you need to specify all paths starting with $ to your local paths

# disrupt mode: constant, tidal
disrupt_mode = "tidal"

# adjustable model parameters (run mode)
p2 = 7.0
p3 = 1.0
kappa = 1.5

# grid length in calculating tidal tensor, in kpc
d_tid = 0.1

# x and y parameters if disrupt_mode == 'tidal'
disrupt_x = 0.67
disrupt_y = 1.33

# Normalized period of rotation for disruption if disrupt_mode == 'constant'
pr = 0.5

# log(Mcut) - knee of the Schechter function. Anything with log_mc >
# 10 will be treated as a pure power-law, to speed up runtime
# (analytic solution).
log_mc = 7.0

seed = 0  # random seed

# Output file names: will result in cat_base_mcvalue_p2_p3.txt ,
# allcat_base_mcvalue_p2_p3.txt, merits_base.txt
allcat_base = "allcat"
merit_name = "merits.txt"

# Optional model parameters. These are usually taken as fixed, but in
# principle can vary
mpb_only = False

# Scaling relations parameters, as in Chen & Gnedin 2023
mmr_slope = 0.3  # Mass slope of galaxy mass-metallicity relation
mmr_pivot = 9.0  # Pivot in the power-law of mass-metallicity relation
mmr_evolution = 1.0  # Redshift slope of the galaxy mass-metallicity relation
mmr0 = -0.5  # The constant to shift MMR
max_feh = 0.3  # Max [Fe/H]

# Scaling of the gas depletion time with redshift, tdep \propto
# (1+z)^(-alpha), alpha as here
tdep = 0.3

sigma_mg = 0.3  # Galactic MMR scatter, in dex [0]
sigma_mc = 0  # Cluster metallicity scatter (within a single galaxy), in dex [0.3]
sigma_gas = 0.3  # Cold gas fraction ratio scatter, in dex

# Turn on stellar mass scatter (only True or False; if True, use the
# Behroozi 2013 has an evolving stellar mass scatter)
sm_scat = True
gaussian_process = True  # whether to evole it with Gaussian process

gauss_l = 2  # correlation length of Gaussian process, in Gyr
gauss_l_sm = 2  # correlation length of Gaussian process for stellar mass, in Gyr

regen_feh = False
seed_feh = 0  # seed for re-generate feh

log_Mhmin = 8.0  # Min halo mass
log_Mmin = 4.0  # Min cluster mass to draw from CIMF

t_lag = 0.01  # time lag for assigning stellar particles, in Gyr

low_mass = False  # whether to impose the low mass GCs < Mmin
log_Mmin_low_mass = 4.0

low_mass_attempt_N = 1  # number of attempts to for clusters if GC Mass is below Mmin

form_nuclear_cluster = True
no_random_at_formation = False

# base_tree = "$base_tree"
base_tree = "data/base_tree/"
base_halo = "data/base_halo/"
# resultspath = "data/result/m12i/raw/it_%d/" % it
resultspath = "data/result/m12i/raw/"

# redshift list
offset = 4
redshift_snap = np.loadtxt("data/external/fire_z_list.txt", dtype=float)[offset:]

# path of mass loss due to stellar evolution
path_massloss = "data/massloss.txt"  # not used

mu_sev = 0.55  # fraction of stellar mass after stellar evolution

rmax_form = 3.0  # max radius to form GCs, in kpc

# subhalo list
subs = [25236877]  # m12i

# Input: color-metallicity transformations to be used for the Virgo
# Cluster GCs. Options right now are "LG14", "CGL18", "V19"
color_metallicity = "CGL18"

UVB_constraint = "KM22"

# output the label whether the GC mass exceeds stellar mass
exceed_stellar = False

# whether to fix stellar mass when the GC mass exceeds stellar mass
fix_stellar = False

skip = None

verbose = True

full_snap_only = True

max_lag_ratio = 1.0

form_nuclear_cluster = True

# Paramaters added by Finn P.

# frac_rs = 1  # look at Chen 2023 Section 2.2.3 Particle assignment
# collisionless_only = True  # not used for now
# assign_at_peaks = True  # not used for now
# evenly_distribute = True  # not used for now
# rmin_peaks_finding = 1  # kpc, (I think) look at Chen 2023 Section 2.2.3 Particle assignment
# peak_radius = 0.5  # kpc, (I think) look at Chen 2023 Section 2.2.3 Particle assignment

# Below: simulation specific parameters

# Cosmological parameters
Om = 0.272
Ob = 0.0455
h100 = 0.702  # TNG50

# full snap list if not all snaps are full
full_snap = np.array(
    [
        20,
        23,
        26,
        29,
        33,
        37,
        41,
        46,
        52,
        59,
        67,
        77,
        88,
        102,
        120,
        142,
        172,
        214,
        277,
        294,
        312,
        332,
        356,
        382,
        412,
        446,
        486,
        534,
        590,
        591,
        592,
        593,
        594,
        595,
        596,
        597,
        598,
        599,
        600,
    ],
    dtype=int,
)

full_snap = full_snap - offset
analyse_snap = full_snap[-1]


params = {
    "disrupt_mode": disrupt_mode,
    "h100": h100,
    "Ob": Ob,
    "Om": Om,
    "p2": p2,
    "p3": p3,
    "kappa": kappa,
    # "p2_arr": p2_arr, # not used in model
    # "p3_arr": p3_arr, # not used in model
    # "kappa_arr": kappa_arr, # not used in model
    "d_tid": d_tid,
    "disrupt_x": disrupt_x,
    "disrupt_y": disrupt_y,
    "log_mc": log_mc,
    "seed": seed,
    # "seed_list": seed_list, # not used in model
    "seed_feh": seed_feh,
    "mpb_only": mpb_only,
    # "test_mmr": test_mmr, # regen_feh set to False so not needed
    "mmr_slope": mmr_slope,
    "mmr_pivot": mmr_pivot,
    "mmr_evolution": mmr_evolution,
    "max_feh": max_feh,
    "mmr0": mmr0,
    "tdep": tdep,
    "sigma_mg": sigma_mg,
    "sigma_mc": sigma_mc,
    "sigma_gas": sigma_gas,
    "sm_scat": sm_scat,
    "log_Mhmin": log_Mhmin,
    "log_Mmin": log_Mmin,
    "pr": pr,
    "t_lag": t_lag,
    "base_tree": base_tree,
    "base_halo": base_halo,
    "redshift_snap": redshift_snap,
    "path_massloss": path_massloss,
    "mu_sev": mu_sev,
    "rmax_form": rmax_form,
    "subs": subs,
    "full_snap": full_snap,
    "color_metallicity": color_metallicity,
    "resultspath": resultspath,
    "allcat_base": allcat_base,
    "merit_name": merit_name,
    "low_mass": low_mass,
    "log_Mmin_low_mass": log_Mmin_low_mass,
    "form_nuclear_cluster": form_nuclear_cluster,
    "low_mass_attempt_N": low_mass_attempt_N,
    "no_random_at_formation": no_random_at_formation,
    "skip": skip,
    "gaussian_process": gaussian_process,
    "gauss_l": gauss_l,
    "gauss_l_sm": gauss_l_sm,
    "regen_feh": regen_feh,
    "UVB_constraint": UVB_constraint,
    "exceed_stellar": exceed_stellar,
    "fix_stellar": fix_stellar,
    "verbose": verbose,
    "analyse_snap": analyse_snap,
    "full_snap_only": full_snap_only,
    "max_lag_ratio": max_lag_ratio,
    # "frac_rs": frac_rs,
    # "collisionless_only": collisionless_only,
    # "assign_at_peaks": assign_at_peaks,
    # "evenly_distribute": evenly_distribute,
    # "rmin_peaks_finding": rmin_peaks_finding,
    # "peak_radius": peak_radius,
}
