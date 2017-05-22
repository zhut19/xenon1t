import sys, os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None        # default='warn'

pax_version = '6.6.5'

def delayed_peak_extraction(run, ifolder = None, ofolder = None):
    dfe = pd.read_pickle('/home/zhut/data/Delayed/data/pickles/pax_v%s_elist_cut/%s.pkl' % (pax_version, run))
    dfp = pd.read_pickle('/home/zhut/data/Delayed/data/pickles/pax_v%s_peak_raw/%s.pkl' % (pax_version, run))
    elist = dfe.event_number.values
    dfe = pd.read_pickle('/home/zhut/data/Delayed/data/pickles/pax_v%s_elist_all/%s.pkl' % (pax_version, run))

    time_coverage = []
    data_cache = []
    
    for i, event in enumerate(elist):
        # First event ############################################
        dfp_ = dfp[dfp.event_number == event]
        ns2 = np.max(dfp_[dfp_.type == 2].area.values)
        ns2_htm = dfp_[dfp_.area == ns2].hit_time_mean.values[0]
        dfp_['time_diff_ns2'] = dfp_.hit_time_mean - ns2_htm
        dfp_['ns2'] = ns2
        cumus2 = np.sum(dfp_[(dfp_.type == 2) & (abs(dfp_.time_diff_ns2) < 300e3)].area.values)
        dfp_['cumus2'] = cumus2
        data_cache.append(dfp_)
        
        
        start = dfe[dfe.event_number == event].event_time.values[0] - ns2_htm
        end = start + dfe[dfe.event_number == event].event_duration.values[0]
        time_coverage.append((start, end, cumus2))
        
        # Second event ############################################
        dfp_ = dfp[dfp.event_number == event+1]

        if len(dfp_) == 0:
            continue

        try:
            first_large_peak = dfp_[dfp_.area>1e3].hit_time_mean.values[0]
        except IndexError:
            first_large_peak = (dfe[dfe.event_number == event+1].event_time.values[0] + 
                                dfe[dfe.event_number == event+1].event_duration.values[0])
            
        dfp_ = dfp_[dfp_.hit_time_mean < first_large_peak]
        dfp_['time_diff_ns2'] = dfp_.hit_time_mean - ns2_htm
        dfp_['ns2'] = ns2
        dfp_['cumus2'] = cumus2 
        data_cache.append(dfp_)
        
        
        start = dfe[dfe.event_number == event+1].event_time.values[0] - ns2_htm
        end = first_large_peak - ns2_htm
        time_coverage.append((start, end, cumus2))
            
    data_cache = pd.concat(data_cache)

    pfile = os.path.join(ofolder,run+'.pkl')
    data_cache.to_pickle(pfile)
    np.savez('/home/zhut/data/Delayed/data/npzs/pax_v%s_peak_combine/%s.npz' %(pax_version,run), np.asarray(time_coverage))

    print ('This Job is Complete :] \nFile %s has been created in %s' % (run,ofolder.split('/')[-1]))

if __name__ == "__main__":
    run = sys.argv[1]
    out_folder = '/home/zhut/data/Delayed/data/pickles/pax_v%s_peak_combine' % pax_version
    delayed_peak_extraction(run, ifolder = None, ofolder = out_folder)
