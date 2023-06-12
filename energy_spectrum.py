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
                    bins=30, line_label='simulation', xlabel=r'E [GeV]', xlim=(1e2,1e6), 
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
    ax.errorbar(center_energy, hist, yerr=y_err, fmt='o-', label=line_label, markersize=3)
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
    return 0.14e4 * Emu**-2.7 * ( 1/(1 + 1.1*Emu*cs/115) + 0.054/(1 + 1.1*Emu*cs/850))

def MUPAGE_flux(m: int, h: float, theta: float):
    """
    m: number of muons in a muon bundle
    h: depth of interests: [km]
    theta: muon direction
    """
    from math import cos, exp
    k0a = 7.2e-3
    k0b = -1.927
    k1a = -0.581
    k1b = 0.034
    v0a = 7.71e-2
    v0b = 0.524
    v0c = 2.068
    v1a = 0.03
    v1b = 0.47

    #Lambda
    Lambda = 0.

    # K(h, theta)    
    k0 = k0a * h**k0b
    k1 = k1a * h + k1b
    k = k0*cos(theta) * exp( k1/cos(theta) ) # = flux of m=1 muon bundle with theta at h. unit: m-2s-1sr-1

    # \nu(h, theta)
    v0 = v0a*h**2 + v0b * h + v0c
    v1 = v1a * exp(v1b*h)
    v = v0 * exp(v1/cos(theta))
    return k / m**v


def parameterized_vertical_muon_energy_spectrum_with_depth(Emu: Union[float, np.array], depth:float, zenith=0):
    # Emu [GeV], depth [km], zenith [rad]
    from math import exp, cos, log
    X = depth / cos(zenith)
    beta, y0, y1, ep0a, ep0b, ep1a, ep1b = 0.42, -0.232, 3.961, 0.0304, 0.359, -0.0077, 0.659
    ep1 = ep1a * depth + ep1b
    ep0 = ep0a * exp(ep0b * depth)
    ep = ep0 / cos(zenith) + ep1
    y = y0 * log(depth) + y1
    G = 2.3 * (y-1) * ep**(y-1) * exp((y-1)*beta*X) * (1-exp(-beta*X))**(y-1)
    return G * Emu/1e3 * exp(beta*X*(1-y)) * (Emu/1e3 + ep * (1-exp(-beta*X)))**(-y)


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

    
