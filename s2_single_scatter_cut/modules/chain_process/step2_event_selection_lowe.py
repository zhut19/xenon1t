# Step2: Make quality cuts in my_lax_copy_sr1
# A bit different from actual lax, instead of adding columns, here just make the cut
# Remove useless columns once the cuts are done.

# Have additional low energy cut
# ** Change fv if using AmBe data

import os, sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None        # default='warn'

sys.path.append('/home/zhut/data/SingleScatter/modules')
from jupyter2module import NotebookLoader
nb = NotebookLoader(['/home/zhut/data/'])  
laxc = nb.load_module('my_lax_copy_sr1')

pax_version = '6.6.5'

def quality_control_cuts(df, q = False):
    
    ie = laxc.InteractionExists()
    s2w = laxc.S2Width()
    ambefv = laxc.AmBeFiducial()
    fv = laxc.FiducialCylinder1T()
    daqv = laxc.DAQVeto()
    s2pll = laxc.S2PatternLikelihood()
    s2aft = laxc.S2AreaFractionTop()
    
    cuts = [ie,fv,s2w,daqv,s2aft]
    
    for cut in cuts:

        df = df[df.cs2 < 1e4]
        df = df[df.cs1 < 5e2]

        n_before = len(df)
        df = cut.process(df)
        n_after = len(df)

        if not q: print ('%.2f%% passed %s' %(n_after/n_before *100 ,cut.name()))
        
    return (df)

def event_selection(run, ifolder = None, ofolder = None):

    pfile = os.path.join(ifolder,run+'.pkl')
    df = pd.read_pickle(pfile)
    cols = df.columns
    not_in_use_col = []
    not_in_use_col += [col for col in cols if ('nearest' in col) or ('next' in col) or ('previous' in col)]
    not_in_use_col += [col for col in cols if ('largest_' in col) or ('alt' in col)]
    not_in_use_col += [col for col in cols if ('pe_event' in col) or ('busy' in col)]
    not_in_use_col += [col for col in cols if ('channels' in col) or ('nn' in col)]
    not_in_use_col += [col for col in cols if ('n_' in col) or ('_80p_' in col) or ('total' in col)]
    keep = ['largest_other_s2','previous_busy_on','previous_busy_off','nearest_busy','nearest_hev',
      'run_number','s1_pattern_fit','s2_pattern_fit','s1_area_fraction_top','s2_area_fraction_top']
    not_in_use_col = [col for col in not_in_use_col if not col in keep]
    further_col_to_dorp = ['previous_busy_on','previous_busy_off','nearest_busy','nearest_hev',
      'r_pos_correction','s1_tight_coincidence','z_pos_correction','event_duration', 'event_time',
      'largest_other_s2']
    
    df.drop(not_in_use_col,axis = 1,inplace = True)
    df = quality_control_cuts(df, q = False)
    df.drop(further_col_to_dorp,axis = 1,inplace = True)

    pfile = os.path.join(ofolder, run+'.pkl')
    df.to_pickle(pfile)
    print ('This Job is Complete :] \nFile %s has been created in %s' % (run,ofolder.split('/')[-1]))

if __name__ == "__main__":
    run = sys.argv[1]
    in_folder = '/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_elist' % pax_version
    out_folder = '/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_elist_cut' % pax_version
    event_selection(run, ifolder = in_folder, ofolder = out_folder)



