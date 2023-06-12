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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
font = {'family': 'serif',
        'weight': 'normal', 'size': 12}
plt.rc('font', **font)

import config as con
con.mcevents_suffix = "mc_events.json"
con.raw_power_index = -1
con.sample_area = pi * 500**2

import energy_spectrum as es
from samples import Samples


def save_samples():
    sample_path_prefix = "/lustre/collider/mocen/project/hailing/data/atm_muon/dataStore/sealevel/"

    samples = Samples()

    # load samples
    print('E1e2-1e3', flush=True)
    # batch_path = glob.glob(sample_path_prefix + "E1e2-1e3_5June2023/separate/batch*")
    batch_path = [sample_path_prefix + f"E1e2-1e3_5June2023/separate/batch{i}" for i in range(10)]
    samples.add_sample(batch_path_list=batch_path, n_simu_events=1e8/10, E_min=1e2, E_max=1e3)
    
    print('E1e3-1e4', flush=True)
    # batch_path = glob.glob(sample_path_prefix + "E1e3-1e4_5June2023/separate/batch*")
    batch_path = [sample_path_prefix + f"E1e3-1e4_5June2023/separate/batch{i}" for i in range(10)]
    samples.add_sample(batch_path_list=batch_path, n_simu_events=1e8/10, E_min=1e3, E_max=1e4)

    print('E1e4-1e5', flush=True)
    # batch_path = glob.glob(sample_path_prefix + "E1e4-1e5_5June2023/separate/batch*")
    batch_path = [sample_path_prefix + f"E1e4-1e5_5June2023/separate/batch{i}" for i in range(10)]
    samples.add_sample(batch_path_list=batch_path, n_simu_events=1e7/10, E_min=1e4, E_max=1e5)

    print('E1e5-1e6', flush=True)
    # batch_path = glob.glob(sample_path_prefix + "E1e5-1e6_5June2023/separate/batch*")
    batch_path = [sample_path_prefix + f"E1e5-1e6_5June2023/separate/batch{i}" for i in range(100)]
    samples.add_sample(batch_path_list=batch_path, n_simu_events=1e6/1, E_min=1e5, E_max=1e6)
    
    print('E1e6-1e7', flush=True)
    batch_path = [sample_path_prefix + f"E1e6-1e7_5June2023/batch{i}" for i in range(50)]
    samples.add_sample(batch_path_list=batch_path, n_simu_events=1e3*50, E_min=1e6, E_max=1e7)
    
    print('E1e7-1e8', flush=True)
    batch_path = [sample_path_prefix + f"E1e7-1e8_5June2023/batch{i}" for i in range(20)]
    samples.add_sample(batch_path_list=batch_path, n_simu_events=100*20, E_min=1e7, E_max=1e8)

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

    if args.reload or (not os.path.isfile('../save/detected_particles.csv')):
        print('reloading sample...', flush=True)
        save_samples()
        print('sample saved', flush=True)

    # get samples
    detected_p = pd.read_csv('../save/detected_particles.csv')
    primary = pd.read_csv('../save/primaries.csv')

    """
    Vertical Muon Energy Spectrum
    """
    # cut on z, costheta and radius
    z_cut, cz_cut, R_cut = 1, 0.88, 400
    muons = detected_p.loc[abs(detected_p.pdgid)==13]
    vdonw_muons = select_sample(df_p=muons, z_cut=z_cut, cz_cut=cz_cut, R_cut=R_cut)

    # setup figure 
    fig = plt.figure(figsize=(6,6), dpi=400)
    grid = plt.GridSpec(4, 1, hspace=0., wspace=0.)
    ax_main = fig.add_subplot(grid[:-1,0])
    ax_ratio = fig.add_subplot(grid[-1,0], sharex=ax_main)
    ax_ratio.yaxis.set_minor_locator(MultipleLocator(0.1))
    

    # energy spectrum
    # raw sample should multiply a reweighting factor: cos(theta_prim)
    showerId = vdonw_muons.showerId.unique()
    prim_cos_z = -primary.loc[showerId, 'pz'] /primary.loc[showerId, 'e0'] 
    prim_cos_z = prim_cos_z.loc[vdonw_muons.showerId].to_numpy()
    vdonw_muons['weight'] = vdonw_muons['weight'] * prim_cos_z

    # dE/dS/dt/dOmega/dE
    S = pi * R_cut**2
    dOmega = math.acos(cz_cut)**2 * pi
    weight_scale_factor = 1. / S / dOmega

    center_energy, hist, y_err, ax_main, _ = es.draw_energy_spectrum(vdonw_muons, weight_scale_factor=weight_scale_factor,
                title='Vertical Muon Energy Spectrum', xlim=(0.7e2,1.3e6), ax=ax_main)
    
    np.save('../save/mc_verticalmuon_EvsFlux.npy', np.concatenate( [center_energy.reshape(-1,1), hist.reshape(-1,1)], axis=1 ) )

    # draw analytical result
    analytical_intensity  = es.intensity_sea_level(center_energy, zenith=0)
    ax_main.plot(center_energy, analytical_intensity, label='analytical', linestyle='--')

    # draw ice cube result
    def icecube_sea_level_vertical_muon_flux(energy):
        # fitting result from icecube with muon cosz>0/88 & muon log(E/GeV) in (3.8 5.6)
        # energy: GeV
        # return: s-1m-2sr-1GeV-1
        return 9e-13*(energy/50000)**(-3.74)
    ic_energy = 10**np.linspace(3.8, 5.6)
    ic_flux = icecube_sea_level_vertical_muon_flux(ic_energy)
    ax_main.plot(ic_energy, ic_flux, label='IceCube fitting result', linestyle='--', zorder=3, color='red')

    ax_main.legend()


    # draw ratio plot
    ax_ratio.axhline(y=1, linestyle=':')

    # mc / analytical
    ratio = hist / analytical_intensity
    ax_ratio.plot(center_energy, ratio, linestyle='solid', label='mc / analytical')

    # IceCube / analytical
    ic_ratio = ic_flux / es.intensity_sea_level(ic_energy, zenith=0)
    ax_ratio.plot(ic_energy, ic_ratio, color='red', linestyle='--', label='IceCube / analytical')

    ax_ratio.set_ylim(0,5)
    # ax_ratio.set_yscale('log')
    ax_ratio.set_ylabel('ratio')
    ax_ratio.set_xlabel('E [GeV]')
    ax_ratio.legend(loc='upper left')
    plt.tight_layout() 

    fig.savefig('../save/energy_spectrum.pdf')


    # """
    # other particle 
    # """
    # otherp = detected_p.loc[abs(detected_p.pdgid)!=13]
    # otherp = select_sample(df_p=otherp, z_cut=z_cut, R_cut=R_cut)


    # # setup figure 
    # fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
    
    # # dE/dS/dt/dE
    # S = pi * R_cut**2
    # weight_scale_factor = 1. / S  
    # bins = np.linspace(np.log10(otherp.e0.min()), np.log10(otherp.e0.max()), 50)

    # pdg_to_name = {211: r'$\pi$', 111: r'$\pi^0$', 2212: 'p', 2112: 'n', 321: 'K',
    #              22: r'$\gamma$', 3122: r'$\Lambda$', 3112: r'$\Sigma$'}
    # for pdgid in (abs(otherp.pdgid)).unique():
    #     particles = otherp.loc[abs(otherp.pdgid)==pdgid]
    #     center_energy, hist, y_err, ax, _ = es.draw_energy_spectrum(particles, weight_scale_factor=weight_scale_factor, bins=bins,
    #             title='Energy Spectrum', xlim=(0.8e2,1.3e4), ax=ax, line_label=pdg_to_name[pdgid], ylabel=r'Intensity $[m^{-2}s^{-1}GeV^{-1}]$',)
    #     ax.set_ylim(1e-30, 1e-18)

    # fig.savefig('../save/particles_other_than_mu.png')
