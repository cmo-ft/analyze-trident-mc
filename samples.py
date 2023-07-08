import pandas as pd
import numpy as np
import uproot as ur
import os
import glob
import json
from pathlib import Path
from typing import Union, List
from scipy.integrate import quad
import math
from math import pi
import gc

import config as con

class Single_Sample:
    def __init__(self, batch_path_list: Union[List[str], List[Path]],
                 n_simu_events: int, E_min: float, E_max: float, particle_type: str='',
                 raw_power_index: float=con.raw_power_index, sample_area: float=con.sample_area,
                 sample_solid_angle: float=con.sample_solid_angle, root_file_suffix: str=None) -> None:
        """
        Merge a set of sample from batch_path_list, and reweight samples.
        The batches should come from the same simulation configuration
        """
        if particle_type !=  '':
            assert particle_type in ['p', 'He', 'CNO', 'Mg', 'Fe']
        self.particle_type = particle_type
        self.raw_power_index = raw_power_index
        self.sample_area = sample_area
        self.sample_solid_angle = sample_solid_angle

        self.batch_path_list = batch_path_list
        self.n_simu_events = n_simu_events
        self.E_min = E_min
        self.E_max = E_max
        self.root_file_suffix = root_file_suffix
        self.pmthits = None
        self.primary, self.detect_p = None, None
        self.get_data()
        

    def get_data(self):        
        merged_primary, merged_detect_p = [], []
        merged_pmthits = []
        global_shower_id = 0
        for batch in self.batch_path_list:
            print(batch)
            with open(Path(batch) / con.mcevents_suffix) as f:
                particles = json.load(f)

            primary = pd.json_normalize(particles, record_path='particles_in', meta=['event_id']).set_index('event_id')
            detect_p = pd.json_normalize(particles, record_path='particles_at_detector', meta=['event_id']).set_index('event_id')
            del particles
            gc.collect()

            # get events after cut
            # mask = (detect_p.x**2 + detect_p.y**2 < con.lateral_radius_cut**2)
            # detect_p = detect_p.loc[mask]
            indexes = detect_p.index.unique()
            primary = primary.loc[indexes]

            # sort index
            index_to_showerId = {indexes[i]: i+global_shower_id for i in range(len(indexes))}
            global_shower_id += len(indexes)
            
            detect_p['showerId'] = detect_p.index.map(index_to_showerId)
            primary['showerId'] = primary.index.map(index_to_showerId)

            # get pmthits
            if self.root_file_suffix != None:
                pmthits = ur.open(os.path.join(Path(batch) / self.root_file_suffix))['PmtHit'].arrays(library='pd').droplevel(1)
                # quantum efficiency
                pmthits = pmthits.loc[np.random.rand(len(pmthits))<0.3]
                pmthits['showerId'] = pmthits.index.map(index_to_showerId)
                merged_pmthits.append(pmthits)

            merged_primary.append(primary)
            merged_detect_p.append(detect_p)
        
        merged_primary = pd.concat(merged_primary).set_index('showerId')
        merged_primary['e0'] = np.linalg.norm(merged_primary[['px','py','pz']], axis=1)

        new_weight = Single_Sample.get_new_weight(merged_primary.e0.to_numpy(), self.E_min, self.E_max, self.n_simu_events, \
            particle_type=self.particle_type)
        merged_primary['weight'] = new_weight

        merged_detect_p = pd.concat(merged_detect_p).set_index('showerId')
        merged_detect_p['e0'] = np.linalg.norm(merged_detect_p[['px','py','pz']], axis=1)
        merged_detect_p['weight'] = merged_primary['weight']

        self.primary, self.detect_p = merged_primary, merged_detect_p

        # pmthits
        if self.root_file_suffix != None:
            merged_pmthits = pd.concat(merged_pmthits).set_index('showerId')
            merged_pmthits['weight'] = merged_primary['weight']
            self.pmthits = merged_pmthits

    def raw_spectrum(self, energy:float):
        # energy spectrum used during simulation
        return pow(energy, self.raw_power_index)

    @staticmethod
    def true_spectrum(energy, particle_type:str=''):
        # treat heavy nucleon as one proton with same energy
        assert isinstance(energy, float) or isinstance(energy, np.ndarray), "x must be a float or a NumPy ndarray"
        # input energy unit: GeV
        yc, epc = -4.7, 1.87
        # gammaz list: p, He, CNO, Mg, Fe
        yz = [-2.71, -2.64, -2.68, -2.67, -2.58]
        phiz = [8.73e-2, 5.71e-2, 3.24e-2, 3.16e-2, 2.18e-2]
        Ez = [4.5e6, 9e6, 3.06e7, 6.48e7, 1.17e8]
        Az = [1, 4, 14, 24, 56]
        all_particle_type = ['p', 'He', 'CNO', 'Mg', 'Fe']
        flux = 0
        if particle_type !=  '':
            assert particle_type in all_particle_type
            idx = all_particle_type.index(particle_type)
            flux = phiz[idx] * (energy/1e3)**yz[idx] * ( 1 + (energy/Ez[idx])**epc )**((yc-yz[idx])/epc) / 1e3
        else:
            for i in range(len(yz)):
                flux += phiz[i] * (energy/1e3)**yz[i] * ( 1 + (energy/Ez[i])**epc )**((yc-yz[i])/epc) / 1e3
        return flux
    # def true_spectrum(energy):
    #     # treat heavy nucleon as A protons with energy E/A
    #     assert isinstance(energy, float) or isinstance(energy, np.ndarray), "x must be a float or a NumPy ndarray"
    #     # input energy unit: GeV
    #     yc, epc = -4.7, 1.87
    #     # gammaz list: p, He, CNO, Mg, Fe
    #     yz = [-2.71, -2.64, -2.68, -2.67, -2.58]
    #     phiz = [8.73e-2, 5.71e-2, 3.24e-2, 3.16e-2, 2.18e-2]
    #     Ez = [4.5e6, 9e6, 3.06e7, 6.48e7, 1.17e8]
    #     Az = [1, 4, 14, 24, 56]
    #     flux = 0
    #     for i in range(len(yz)):
    #         e0 = energy * Az[i]
    #         flux += Az[i] * phiz[i] * ( e0 /1e3)**yz[i] * ( 1 + (e0/Ez[i])**epc )**((yc-yz[i])/epc) / 1e3
    #     return flux

    @classmethod
    def get_new_weight(cls, energy: Union[float, np.ndarray], E_min: float, E_max: float, n_events: int,
                raw_power_index: float=con.raw_power_index, sample_area: float=con.sample_area, sample_solid_angle: float=con.sample_solid_angle, \
                particle_type:str=''):
        assert isinstance(energy, float) or isinstance(energy, np.ndarray), "energy must be a float or a NumPy ndarray"
        
        def raw_spectrum(energy:float):
            return pow(energy, raw_power_index)

        raw_spectrum_norm_factor = 1. / quad(raw_spectrum, E_min, E_max)[0]
        weight = cls.true_spectrum(energy, particle_type) / ( energy**raw_power_index * raw_spectrum_norm_factor * n_events) * sample_area * sample_solid_angle
        return  weight
    



class Samples:
    def __init__(self, root_file_suffix: str=None,) -> None:
        """
        Get a list of samples with different simulation configuration.
        Samples should be added from cls.add_sample
        :param root_file_suffix: the path of data.root relative to each batch. e.g.:detectorResponse/data.root
        """
        self.sample_list: List[Single_Sample] = []
        self.root_file_suffix = root_file_suffix
        
    def add_sample(self, batch_path_list: Union[List[str], List[Path]],
                   n_simu_events: int, E_min: float, E_max: float, particle_type: str='',
                   raw_power_index: float=con.raw_power_index, sample_area: float=con.sample_area,
                   sample_solid_angle: float=con.sample_solid_angle):
        self.sample_list.append(Single_Sample(batch_path_list, n_simu_events, E_min, E_max,
                                              particle_type, raw_power_index, sample_area,
                                              sample_solid_angle, root_file_suffix=self.root_file_suffix))
    
    def get_sample_list(self):
        return self.sample_list
    
    @property
    def primary(self):
        if not (hasattr(self, "_primary") or hasattr(self, "_detect_p")):
            self.merge_sample_list()
        return self._primary

    @property
    def detect_p(self):
        if not (hasattr(self, "_primary") or hasattr(self, "_detect_p")):
            self.merge_sample_list()
        return self._detect_p
    
    def merge_sample_list(self):
        global_shower_id = 0
        reweighted_primary, reweighted_detect_p = [], []
        for sample in self.sample_list:
            primary, detect_p = sample.primary.copy().reset_index(0), sample.detect_p.copy().reset_index(0)

            indexes = primary.showerId
            # reassign index
            index_to_showerId = {indexes[i]: i+global_shower_id for i in range(len(indexes))}
            global_shower_id += len(indexes)

            primary['showerId'] = primary.showerId.map(index_to_showerId)
            detect_p['showerId'] = detect_p.showerId.map(index_to_showerId)
            reweighted_primary.append(primary)
            reweighted_detect_p.append(detect_p)
        self._primary, self._detect_p = pd.concat(reweighted_primary).set_index('showerId'), pd.concat(reweighted_detect_p).set_index('showerId')

    def merge_pmt_hits(self):
        global_shower_id = 0
        reweighted_pmthits = []
        for sample in self.sample_list:
            pmthits = sample.pmthits.copy().reset_index(0)

            indexes = pmthits.showerId
            # reassign index
            index_to_showerId = {indexes[i]: i+global_shower_id for i in range(len(indexes))}
            global_shower_id += len(indexes)

            pmthits['showerId'] = pmthits.showerId.map(index_to_showerId)
            reweighted_pmthits.append(pmthits)
        self._pmthits = pd.concat(reweighted_pmthits).set_index('showerId')

    def save_samples(self, target_dir: Union[str, Path]):
        if not (hasattr(self, "_primary") or hasattr(self, "_detect_p")):
            self.merge_sample_list()
        self._primary.to_csv(Path(target_dir) / 'primaries.csv', index_label='showerId')
        self._detect_p.to_csv(Path(target_dir) / 'detected_particles.csv', index_label='showerId')

        if (self.root_file_suffix!=None):
            if (not hasattr(self, "_pmthits")):
                self.merge_pmt_hits()
            self._pmthits.to_csv(Path(target_dir) / 'pmthits.csv', index_label='showerId')

    def __getitem__(self, index):
        return self.sample_list[index]

    def __len__(self):
        return  len(self.sample_list)



# example
if __name__=='__main__':
    sample_path_prefix = "/lustre/collider/mocen/project/hailing/data/atm_muon/dataStore/sealevel/"
    con.raw_power_index = -1
    con.sample_area = pi * 400**2

    samples = Samples()

    # load samples
    batch_path = glob.glob(sample_path_prefix + "test/batch*")
    samples.add_sample(batch_path_list=batch_path, n_simu_events=2e5, E_min=1e2, E_max=1e3)
    samples.save_samples('./example')
    print(samples.primary)
    print(samples.detect_p)