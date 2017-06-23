# Linking all the data processing steps together, so that all steps will be done in order
# Check function provide information about which step it stops

import sys, os
import pandas as pd

sys.path.append(os.getcwd())

import step1_convert_2_pickle as convert_2_pickle
import step2_event_selection_lowe as event_selection
import step3_make_minitree_v1 as make_minitree
import step4_combine_pickle as combine_pickle

class data_process():

    data_folder = '/home/zhut/data/SingleScatter/data'
    
    input_path = [None,
                  data_folder + '/pickles/pax_v6.6.5_rn_elist_all',
                  data_folder + '/pickles/pax_v6.6.5_rn_elist_cut',
                  [data_folder + '/pickles/pax_v6.6.5_rn_elist_cut',
                   data_folder + '/pickles/pax_v6.6.5_rn_event_ss']
                 ]

    output_path = [
        data_folder + '/pickles/pax_v6.6.5_rn_elist_all',
        data_folder + '/pickles/pax_v6.6.5_rn_elist_cut',
        data_folder + '/pickles/pax_v6.6.5_rn_event_ss',
        data_folder + '/pickles/pax_v6.6.5_rn_event_combine'
                  ]

    for opath in output_path:
        if not os.path.exists(opath):
            os.makedirs(opath)

    def __init__(self, run):
        self.run = run

    def process(self, overwrite = False):
        """
        Go through all the steps that didn't find the run in corresponding output_folder
        """
        flag, path = self.check()
        
        starting_point = len(self.output_path)
        for i, folder in enumerate(self.output_path):
            if path in folder:
                starting_point = i
        if flag: starting_point = -1
        if overwrite: starting_point = 0
        if starting_point != -1:
            processes = [
                convert_2_pickle.load_minitrees,
                event_selection.event_selection,
                make_minitree.make_ss_minitree,
                combine_pickle.combine
                        ]
            convert_2_pickle.hax_init()
            for i, proc in enumerate(processes[starting_point:]):
                index = starting_point+i
                proc(self.run, ifolder = self.input_path[index], ofolder = self.output_path[index])        

    def check(self):
        """
        Check if run exists in the final output folder, if yes return True
        Then check if run exists in the output folder in step order return False and which folder
        """
        name = self.run + '.pkl'
        
        if os.path.isfile(os.path.join(self.output_path[-1],name)):
            return True, ''
        
        for i, path in enumerate(self.output_path):
            if os.path.isfile(os.path.join(path,name)):
                pass
            else:
                return False, path
        
        return True, ''

if __name__ == "__main__":
    run = sys.argv[1]
    dp = data_process(run)
    flag, path = dp.check()
    if flag:
        print ('%s all processed' %run)
    else:
        print ('%s not yet in %s' %(run, path.split('/')[-1])) 

    if not flag:
        dp.process(overwrite = False)