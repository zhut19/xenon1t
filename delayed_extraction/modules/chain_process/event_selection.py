import os, sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None        # default='warn'

def quality_control_cuts(df):
    
    def ses2_size_cut_ori(s2_area):
        linear_0 = s2_area *0.01 + 90
        linear_1 = s2_area * 0.025 + 766
        fermi_dirac_coef_0 = 1 / (np.exp((s2_area - 26000) * 6e-4) + 1) 
        fermi_dirac_coef_1 = 1 / (np.exp((26000 - s2_area) * 6e-4) + 1)
        return linear_0*fermi_dirac_coef_0+linear_1*fermi_dirac_coef_1

    sel_dtime = lambda df:df[(df.event_number >= 10) &
                             (df.previous_event > 10e6) &
                             (df.previous_hev > 50e6) & (df.previous_muon_veto_trigger > 50e6) &
                             (np.log10(df.previous_s2_area**0.25*np.exp(-df.previous_event/5e6)) < -3)
                            ]
    df = sel_dtime(df)
    
    sel_usual = lambda df:df[(df.drift_time > 5e3) &
                             (df.largest_other_s2 < ses2_size_cut_ori(df.s2)) &
                             (df.x ** 2 + df.y **2 < 2200)
                            ]
    df = sel_usual(df)
    # just applying a loose fv cuts

    df['s2aft_up_lim'] = (0.6177399420527526 + 3.713166211522462e-08 * df.s2 + 0.5460484265254656 / np.log(df.s2))
    df['s2aft_low_lim'] = (0.6648160611018054 - 2.590402853814859e-07 * df.s2 - 0.8531029789184852 / np.log(df.s2))
    df = df[df.s2_area_fraction_top < df.s2aft_up_lim]
    df = df[df.s2_area_fraction_top > df.s2aft_low_lim]

    sel_tnext = lambda df:df[(df.next_hev_on > df.next_event) &
                             (df.next_muon_veto_trigger > df.next_event)
                            ]
    
    
    return (df)

def cut_event_pickle(run, ifolder = None, ofolder = None):

    pfile = os.path.join(ifolder,run+'.pkl')
    df = pd.read_pickle(pfile)

    df = quality_control_cuts(df)
    
    pfile = os.path.join(ofolder, run+'.pkl')
    df.to_pickle(pfile)
    print ('This Job is Complete :] \nFile %s has been created in %s' % (run,ofolder.split('/')[-1]))

if __name__ == "__main__":
    run = sys.argv[1]
    hax_init()
    in_folder = '/home/zhut/data/Delayed/data/pickles/pax_v%s_elist_all' % pax_version
    out_folder = '/home/zhut/data/Delayed/data/pickles/pax_v%s_elist_cut' % pax_version
    make_minitree(run, ifolder = in_folder, ofolder = out_folder)



