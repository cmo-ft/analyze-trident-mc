import sys
sys.path.append("/lustre/collider/mocen/project/hailing/analysis/atmos_muon/analyze-trident-mc/")

import os
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import math
from math import pi, cos
import glob
import argparse

import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
font = {'family': 'serif',
        'weight': 'normal', 'size': 12}
plt.rc('font', **font)

import energy_spectrum as es
from samples import Samples
import config as con

def save_samples():
    sample_path_prefix = "/lustre/collider/mocen/project/hailing/data/atm_muon/dataStore/sealevel/"
    con.raw_power_index = -1
    con.sample_area = pi * 400**2

    samples = Samples()

    # load samples
    batch_path = glob.glob(sample_path_prefix + "test/batch*")
    samples.add_sample(batch_path_list=batch_path, n_simu_events=2e5, E_min=1e2, E_max=1e3)
    samples.save_samples('../save')


def select_sample(df_p, z_cut=None, cz_cut=None, R_cut=None):
    mask = pd.Series(data=True, index=df_p.index)

    # upper surface sample
    if z_cut != None:
        mask = mask & (df_p.z==z_cut)
    # df_p = df_p.loc[df_p.z==1].copy()

    # cosz cut
    if cz_cut != None:
        coszenith = -df_p.pz / df_p.e0
        mask = mask & (coszenith>cz_cut)

    # radius smaller than R
    if R_cut != None:
        mask = mask & (df_p.x**2 + df_p.y**2 < R_cut**2)
    
    # return df_p.loc[mask]
    return df_p if (mask is None) else df_p.loc[mask]



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Parameters for draw_energy_spectrum")
    parser.add_argument('-r', '--reload',default=False, action="store_true", help="reload and save sample to ../save")
    args = parser.parse_args()

    if args.reload:
        print('reloading sample...', flush=True)
        save_samples()
        print('sample saved', flush=True)

    # get samples
    detected_p = pd.read_csv('../save/detected_particles.csv')

    """
    Vertical Muon Energy Spectrum
    """
    # cut on z, costheta and radius
    z_cut, cz_cut, R_cut = 1, 0.99, 400
    muons = detected_p.loc[abs(detected_p.pdgid)==13]
    vdonw_muons = select_sample(df_p=muons, z_cut=z_cut, cz_cut=cz_cut, R_cut=R_cut)

    # setup figure 
    fig = plt.figure(figsize=(6,6), dpi=400)
    grid = plt.GridSpec(4, 1, hspace=0., wspace=0.)
    ax_main = fig.add_subplot(grid[:-1,0])
    ax_ratio = fig.add_subplot(grid[-1,0], sharex=ax_main)
    
    # energy spectrum
    # dE/dS/dt/dOmega/dE
    S = pi * R_cut**2
    dOmega = math.acos(cz_cut)**2 * pi
    weight_scale_factor = 1. / S / dOmega 
    center_energy, hist, y_err, ax_main, _ = es.draw_energy_spectrum(vdonw_muons, weight_scale_factor=weight_scale_factor,
                title='Vertical Muon Energy Spectrum', xlim=(0.8e2,1.3e4), ax=ax_main)

    # draw analytical result
    analytical_intensity  = es.intensity_sea_level(center_energy, zenith=0)
    ax_main.plot(center_energy, analytical_intensity, label='analytical', linestyle='-')
    ax_main.legend()

    # draw ratio plot
    ratio = hist / analytical_intensity
    ax_ratio.plot(center_energy, ratio, linestyle='solid')
    ax_ratio.axhline(y=1, linestyle='--')
    ax_ratio.set_ylim(0,2)
    ax_ratio.set_ylabel('mc / analytical')
    ax_ratio.set_xlabel('E [GeV]')
    plt.tight_layout() 

    fig.savefig('../save/energy_spectrum.png')


    """
    other particle 
    """
    otherp = detected_p.loc[abs(detected_p.pdgid)!=13]
    otherp = select_sample(df_p=otherp, z_cut=z_cut, R_cut=R_cut)

    # setup figure 
    fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
    
    # dE/dS/dt/dE
    S = pi * R_cut**2
    weight_scale_factor = 1. / S  
    bins = np.linspace(np.log10(otherp.e0.min()), np.log10(otherp.e0.max()), 50)

    pdg_to_name = {211: r'$\pi$', 111: r'$\pi^0$', 2212: 'p', 2112: 'n'}
    for pdgid in (abs(otherp.pdgid)).unique():
        particles = otherp.loc[abs(otherp.pdgid)==pdgid]
        center_energy, hist, y_err, ax, _ = es.draw_energy_spectrum(particles, weight_scale_factor=weight_scale_factor, bins=bins,
                title='Energy Spectrum', xlim=(0.8e2,1.3e4), ax=ax, line_label=pdg_to_name[pdgid], ylabel=r'Intensity $[m^{-2}s^{-1}GeV^{-1}]$',)
        ax.set_ylim(1e-30, 1e-18)

    fig.savefig('../save/particles_other_than_mu.png')
