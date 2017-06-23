# Step 1 in process chain: Load and combine public minitrees
# Basics,Fundamentals,Proximity,TotalProperties,Extended

# Init hax here in the first step.

import pandas as pd

import sys,os

import hax
from hax.minitrees import TreeMaker
from hax.treemakers.common import Fundamentals
from hax.treemakers.common import Basics
from hax.treemakers.DoubleScatter import DoubleScatter
from hax.treemakers.common import LargestPeakProperties
from hax.treemakers.common import Extended
from hax.treemakers.common import TotalProperties
from hax.treemakers.trigger import Proximity
from hax.treemakers.trigger import TailCut

pax_version = '6.6.5'
data_path = '/project/lgrandi/xenon1t/processed/pax_v%s' % pax_version
your_own_path = '/home/zhut/data/SingleScatter/data/minitrees/pax_v%s_ambe' % pax_version
public_path = '/project2/lgrandi/xenon1t/minitrees/pax_v%s' % pax_version

def hax_init():
    # Trick learned from Daniel Coderre's DAQVeto lichen
    if not len(hax.config):
        # User didn't init hax yet... let's do it now
        hax.init(experiment = 'XENON1T',
                 main_data_paths = [data_path],
                 minitree_paths = [your_own_path,public_path])

def load_minitrees(run, ifolder = None, ofolder = None):
    df = hax.minitrees.load(run,[Basics,Fundamentals,Proximity,TotalProperties,Extended]
                           )
    pfile = os.path.join(ofolder,run+'.pkl')
    df.to_pickle(pfile)
    print ('This Job is Complete :] \nFile %s has been created in %s' % (run,ofolder.split('/')[-1]))

    del df

if __name__ == "__main__":
    run = sys.argv[1]
    hax_init()
    out_folder = '/home/zhut/data/SingleScatter/data/pickles/pax_v%s_ambe_elist' % pax_version
    load_minitrees(run, ofolder = out_folder)