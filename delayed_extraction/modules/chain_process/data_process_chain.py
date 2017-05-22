import sys, os

sys.path.append(os.getcwd())

import convert_2_pickle
import event_selection
import after_sel_peak_treebuilder
import after_sel_delayed_peak

class data_process():

    def __init__(self, run):
        self.run = run

    data_folder = '/home/zhut/data/Delayed/data'

    input_path = [None,
                  data_folder + '/pickles/pax_v6.6.5_elist_all',
                  data_folder + '/pickles/pax_v6.6.5_elist_cut',
                  None,
                 ]

    output_path = [
        data_folder + '/pickles/pax_v6.6.5_elist_all',
        data_folder + '/pickles/pax_v6.6.5_elist_cut',
        data_folder + '/pickles/pax_v6.6.5_peak_raw',
        data_folder + '/pickles/pax_v6.6.5_peak_combine'
                  ]

    def process(self, overwrite = False):
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
                event_selection.cut_event_pickle,
                after_sel_peak_treebuilder.make_peak_minitree,
                after_sel_delayed_peak.delayed_peak_extraction
                        ]
            convert_2_pickle.hax_init()
            for i, proc in enumerate(processes[starting_point:]):
                index = starting_point+i
                proc(self.run, ifolder = self.input_path[index], ofolder = self.output_path[index])        

    def check(self):
        name = self.run + '.pkl'
        
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
    

