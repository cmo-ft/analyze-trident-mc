import os
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import math
from math import pi, cos

import matplotlib.pyplot as plt
font = {'family': 'serif',
        'weight': 'normal', 'size': 12}
plt.rc('font', **font)

import config as con
from samples import Single_Sample, Samples

def draw_energy_spectrum(data: pd.DataFrame, weight_scale_factor: float, 
                    bins=50, line_label='simulation', xlabel=r'E [GeV]', xlim=(1e2,1e6), 
                    ylabel=r'Intensity $[m^{-2}s^{-1}sr^{-1}GeV^{-1}]$',
                    title='energy spectrum', ax=None, fig=None,
                    ):
    # draw dN/dEdt*weight_scale_factor
    spectrum_hist, bin_edges = np.histogram(np.log10(data.e0), bins=bins, weights=data.weight / data.e0 )
    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    center_energy = np.power(10, bin_center)

    # spectrum_hist = dN / (E*dlog10E*dt). transform it to dN/(E*dlnE*dt) = dN/(dEdt)
    hist = spectrum_hist / np.diff(bin_edges)  * np.log10(math.e)

    # and multiply scale factor
    hist *= weight_scale_factor

    # get counting error
    counts, _ = np.histogram(np.log10(data.e0), bins=bins )
    counts[counts==0] = 1
    rel_uncertainty = 1/np.sqrt(counts)
    y_err = hist * rel_uncertainty

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
    ax.errorbar(center_energy, hist, yerr=y_err, fmt='o-', label=line_label)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout() 
    
    return (center_energy, hist, y_err, ax, fig)
    
def intensity_sea_level(Emu: Union[float, np.array], zenith: float):
    """
    Emu in GeV, zenith in rad
    return: dN/(dE dOmega dS dt) with unit 1/(m2 s sr GeV)
    """
    cs = cos(zenith)
    return 0.14e-4 * Emu**-2.7 * ( 1/(1 + 1.1*Emu*cs/115) + 0.054/(1 + 1.1*Emu*cs/850))


if __name__=='__main__':
    # get samples
    detected_p = pd.read_csv('example/detected_particles.csv')

    # select muons
    mask = (abs(detected_p.pdgid)==13)

    # upper surface sample
    mask = mask & (detected_p.z==1)
    # detected_p = detected_p.loc[detected_p.z==1].copy()

    # vertically down-going 
    coszenith = -detected_p.pz / detected_p.e0
    cz_cut = 0.99
    mask = mask & (coszenith>cz_cut)

    # radius smaller than R
    R_cut = 400 #m
    mask = mask & (detected_p.x**2 + detected_p.y**2 < R_cut**2)
    

    # apply cuts
    vdonw_muons = detected_p.loc[mask]


    # dE/dS/dt/dOmega/dE
    S = pi * R_cut**2
    dOmega = math.acos(cz_cut)**2 * pi
    weight_scale_factor = 1. / S / dOmega 
    center_energy, hist, y_err, ax, fig = draw_energy_spectrum(vdonw_muons, weight_scale_factor=weight_scale_factor, xlim=(1e2,1e4))

    analytical_intensity  = intensity_sea_level(center_energy, 0)
    ax.plot(center_energy, analytical_intensity, label='analytical', linestyle='--')
    ax.legend()
    plt.show()

    
