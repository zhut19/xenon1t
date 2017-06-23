# Step 4 in process chain: Combine normal minitrees (dataframe) with OtherLargeS2s minitree (dataframe)

import sys, os
import pandas as pd

pax_version = '6.6.5'

def combine(run, ifolder = None, ofolder = None):
    
    if isinstance(ifolder, list):
        pfile = [os.path.join(f,run+'.pkl') for f in ifolder]
        df = [pd.read_pickle(p) for p in pfile]
        df[1].drop(['run_number','event_number'], axis = 1,inplace = True)
        df[1].index = df[0].index
        df = pd.concat(df, join='outer',axis = 1,verify_integrity = True)
        
    pfile = os.path.join(ofolder,run+'.pkl')
    df.to_pickle(pfile)
    
    print ('This Job is Complete :]')


if __name__ == "__main__":
    run = sys.argv[1]
    in_folder = ['/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_elist_cut' % pax_version,
               '/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_event_ss' % pax_version]
    out_folder = '/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_event_combine' % pax_version
    combine(run, ifolder = in_folder, ofolder = out_folder)