# Step 3 in the chain process: Make OtherLargeS2s minitree
# By using hax.minitrees.load_single_dataset, you don't actually make a .root minitree
# instead you get a dataframe in return and pickle it to where you want

# Updated to get interior_split_fraction branch

import sys, os
import numpy as np
import pandas as pd
import hax
from pax import units, configuration, datastructure

pax_config = configuration.load_configuration('XENON1T')

v_drift = pax_config['DEFAULT']['drift_velocity_liquid']
drift_time_gate = pax_config['DEFAULT']['drift_time_gate']
sample_duration = pax_config['DEFAULT']['sample_duration']
electron_life_time = pax_config['DEFAULT']['electron_lifetime_liquid']

pax_version = '6.6.5'
data_path = '/project/lgrandi/xenon1t/processed/pax_v%s' % pax_version
your_own_path= '/home/zhut/data/SingleScatter/data/minitrees/pax_v'+pax_version
public_path = '/project2/lgrandi/xenon1t/minitrees/pax_v'+pax_version

def hax_init():
    # Trick learned from Daniel Coderre's DAQVeto lichen
    if not len(hax.config):
        hax.init(experiment = 'XENON1T',
                 main_data_paths = [data_path],
                 minitree_paths = [your_own_path,public_path])

class OtherLargeS2s(hax.minitrees.TreeMaker):
    """Information on the large s2 peaks other than the interaction peak. (e.g. s2_1_area)
    Provides:
     - s2_x_y

    x denotes the order of the area of the peak:
     - 1: The 1st largest among peaks other than main interaction peak
     - 2: The 2nd largest among peaks other than main interaction peak
     ...
     - 5: The 5th largest among peaks other than main interaction peak
    
    y denotes the property of the peak:
     - area: The uncorrected area in pe of this peak
     - range_50p_area: The width, duration of region that contains 50% of the area of the peak
     - area_fraction_top: The fraction of uncorrected area seen by the top array
     - x: The x-position of this peak (by TopPatternFit)
     - y: The y-position of this peak
     - z: The z-position of this peak (computed using configured drift velocity)
     - corrected_area: The corrected area in pe of the peak
     - delay_is1: The hit time mean minus main s1 hit time mean
     - delay_is2: The hit time mean minus main s2 hit time mean
     - *interior_split_fraction: Area fraction of the smallest of the two halves considered in the best split inside the peak
     - *goodness_of_fit: Goodness-of-fit of hitpattern to position provided by PosRecTopPatternFit 
    
    Notes:
     - 'largest' refers to uncorrected area, also excluding main interation peak
     - 'z' calculated from the time delay from main interaction s1
     - 'main interaction' is event.interactions[0]
                          (currently is largest S2 + largest S1 before it)
     - 'corrected_area' only corrected by lifetime and spatial 
        
    """
    extra_branches = ['peaks.*']
    peak_name = ['s2_%s_' % order for order in ['1','2','3','4','5']]
    peak_fields = ['area', 'range_50p_area', 'area_fraction_top', 'x', 'y', 'z', 'goodness_of_fit', 
                   'corrected_area', 'delay_is1', 'delay_is2','interior_split_fraction']
    __version__ = '0.1.1'
    
    def extract_data(self, event):
        event_data = dict()
        
        # At least one s1 is needed to anchor all s2s
        if len(event.interactions) != 0:
            # find out which peak are in main interaction
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]
            other_s2s = [ix for ix in event.s2s 
                         if (ix != interaction.s2) and 
                            (event.peaks[ix].index_of_maximum - s1.index_of_maximum > 0)]
            
            # Start looking for the properties we want
            for order, ix in enumerate(other_s2s):
                peak = event.peaks[ix]
                if order >= 5:
                    break
                _current_peak = {}
                
                drift_time = (peak.index_of_maximum - s1.index_of_maximum) * sample_duration
                
                for field in self.peak_fields: # assuming peaks are already sorted
                    # Deal with special cases
                    if field == 'range_50p_area':
                        _x = list(peak.range_area_decile)[5]
                    elif field in ('x', 'y', 'goodness_of_fit'):
                        # In case of x and y need to get position from reconstructed_positions
                        for rp in peak.reconstructed_positions:
                            if rp.algorithm == 'PosRecTopPatternFit':
                                _x = getattr(rp, field)
                                break
                            else:
                                _x = float('nan')
                    elif field == 'z':
                        _x = (- drift_time + drift_time_gate) * v_drift
                    elif field == 'corrected_area':
                        s2_lifetime_correction = np.exp(drift_time / electron_life_time)
                        _x = peak.area * s2_lifetime_correction * peak.s2_spatial_correction
                    elif field == 'delay_is1':
                        _x = (peak.hit_time_mean - s1.hit_time_mean)
                    elif field == 'delay_is2':
                        _x = (peak.hit_time_mean - s2.hit_time_mean)
                    else:
                        _x = getattr(peak, field)
                    
                    field = self.peak_name[order] + field
                    _current_peak[field] = _x

                event_data.update(_current_peak)
         
        return event_data

def make_ss_minitree(run, ifolder = None, ofolder = None):

    pfile = os.path.join(ifolder,run+'.pkl')
    df = pd.read_pickle(pfile)

    elist = df.event_number.values
    
    df, cache = hax.minitrees.load_single_dataset(run,[OtherLargeS2s],event_list = elist)
    
    pfile = os.path.join(ofolder,run+'.pkl')
    df.to_pickle(pfile)
    
    print ('This Job is Complete :]')


if __name__ == "__main__":
    run = sys.argv[1]
    hax_init()
    in_folder = '/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_elist_cut' % pax_version 
    out_folder = '/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_event_ss' % pax_version
    make_ss_minitree(run, ifolder = in_folder, ofolder = out_folder)