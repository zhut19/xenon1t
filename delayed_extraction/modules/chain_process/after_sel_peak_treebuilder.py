import sys, os
import hax
import root_numpy
import numpy as np
import argparse as ap
import pandas as pd
from hax.minitrees import MultipleRowExtractor

pax_version = '6.6.5'
data_path = '/project/lgrandi/xenon1t/processed/pax_v%s' % pax_version
your_own_path = '/home/zhut/data/Delayed/data/minitrees/pax_v%s' % pax_version
public_path = '/project2/lgrandi/xenon1t/minitrees/pax_v%s' % pax_version


def hax_init():
    hax.init(experiment = 'XENON1T',
             main_data_paths = [data_path],
             minitree_paths = [your_own_path,public_path])

class DelayedSingleElectron(MultipleRowExtractor):
    # Using MultipleRowExtractor to get peak as well as event level information. 
    # Also based on peak_treemakers.py by Jelle
    __version__ = '0.0.0'
    
    # Default branch selection is EVERYTHING in peaks, overwrite for speed increase
    # Don't forget to include branches used in cuts
    extra_branches = ['peaks.*']
    uses_arrays = False
    stop_after = np.inf
    
    peak_fields = ['area','type',
                   'range_50p_area','hit_time_mean','area_fraction_top','x','y']
    event_cut_list = []
    peak_cut_list = ['type != "lone_hit"', 'type != "unknown"', 'detector == "tpc"']

    event_cut_string = 'True'
    peak_cut_string = 'True'
    
    # Hacks for want of string support :'(
    peaktypes = dict(lone_hit=0, s1=1, s2=2, unknown=3)
    detectors = dict(tpc=0, veto=1, sum_wv=2, busy_on=3, busy_off=4)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_cut_string = self.build_cut_string(self.peak_cut_list, 'peak')          
    
    def build_cut_string(self, cut_list, obj):
        # If no cut is specified, always pass cut
        if len(cut_list) == 0:
            return 'True'

        cut_string = '('
        for cut in cut_list[:-1]:
            cut_string += obj + '.' + cut + ') & ('
        cut_string += obj + '.' + cut_list[-1] + ')'
        return cut_string

    def extract_data(self, event):
        global elist
        
        if event.event_number == self.stop_after:
            raise hax.paxroot.StopEventLoop()
        peak_data = []
        # Check if in there is an interaction
        if len(event.interactions) >= -1: # what a stupid mistake
            event_data = dict()
            
            edata = ap.Namespace(**event_data)
            
            # Check if event passes cut
            if (eval(self.build_cut_string(self.event_cut_list, 'edata')) and
                ((event.event_number in elist) or (event.event_number in elist + 1))
               ):
                # Loop over peaks and check if peak passes cut

                for i, peak in enumerate (event.peaks):
                    
                    if eval(self.build_cut_string(self.peak_cut_list, 'peak')):
                        # Loop over properties and add them to _current_peak one by one
                        #if peak.type == 's1':
                        

                        _current_peak = {}
                        for field in self.peak_fields:
                            # Deal with special cases
                            if field == 'range_50p_area':
                                _x = list(peak.range_area_decile)[5]
                            elif field == 'rise_time':
                                _x = -peak.area_decile_from_midpoint[1]
                            elif field == 'type':
                                _x = self.peaktypes.get(peak.type, -1)
                            elif field == 'detector':
                                _x = self.detectors.get(peak.detector, -1)
                            elif field == 'hit_time_mean':
                                _x = peak.hit_time_mean + event.start_time
                            elif field in ('x', 'y'):
                                # In case of x and y need to get position from reconstructed_positions
                                for rp in peak.reconstructed_positions:
                                    if rp.algorithm == 'PosRecTopPatternFit':
                                        _x = getattr(rp, field)
                                        break
                                else:
                                    _x = float('nan')
                                # Change field name!
                                field = field + '_peak'
                            else:
                                _x = getattr(peak, field) 
                            
                            _current_peak[field] = _x
                        
                        _current_peak.update(event_data)
                        peak_data.append(_current_peak)
                
                return peak_data
            else:
                return []
        else:
            return []

    def process_event(self, event):
        result = self.extract_data(event)
        if not isinstance(result, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))
        # Add the run and event number to the result. This is required to make joins succeed later on.
        for i in range(len(result)):
            result[i]['run_number'] = self.run_number
            result[i]['event_number'] = event.event_number
        assert len(result) == 0 or isinstance(result[0], dict)
        self.cache.extend (result)
        self.check_cache(force_empty=False)

def make_peak_minitree(run, ifolder = None, ofolder = None):

    pfile = os.path.join(ifolder,run+'.pkl')
    
    global elist
    elist = pd.read_pickle(pfile).event_number.values
    df, cuthistory = hax.minitrees.load_single_dataset(run,DelayedSingleElectron)

    pfile = os.path.join(ofolder,run+'.pkl')
    df.to_pickle(pfile)
    print ('This Job is Complete :] \nFile %s has been created in %s' % (run,ofolder.split('/')[-1]))

    del elist, df

if __name__ == "__main__":
    run = sys.argv[1]
    hax_init()
    in_folder = '/home/zhut/data/Delayed/data/pickles/pax_v%s_elist_cut' % pax_version
    out_folder = '/home/zhut/data/Delayed/data/pickles/pax_v%s_elist_peak' % pax_version
    make_peak_minitree(run, ifolder = in_folder, ofolder = out_folder)
    














    