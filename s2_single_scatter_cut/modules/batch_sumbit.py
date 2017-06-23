# 2016-09, sanderb@nikhef.nl: from Chris with Sander's personalization

x = """#!/bin/bash
#SBATCH --job-name={run}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --output=/home/zhut/log/minitree_%J.log
#SBATCH --error=/home/zhut/log/minitree_%J.log
#SBATCH --account=pi-lgrandi
#SBATCH --partition=kicp
#SBATCH --qos=xenon1t-kicp
export PATH=/project/lgrandi/anaconda3/bin:$PATH
export PROCESSING_DIR=/home/zhut/tmp/pax_v6.6.5_tmp/minitree_{run} 

mkdir -p ${{PROCESSING_DIR}}
cd ${{PROCESSING_DIR}}

source activate pax_head

python /home/zhut/data/SingleScatter/modules/chain_process/data_process_chain.py {run}

rm -rf ${{PROCESSING_DIR}}
"""

# Use submit procedure from CAX
from cax.qsub import submit_job

import os, sys
import time
import pandas as pd
sys.path.append('/home/zhut/data/SingleScatter/modules/chain_process')
from data_process_chain import data_process

# check my jobs
def working_job():
    cmd="squeue --user=zhut | wc -l"
    jobNum=int(os.popen(cmd).read())
    return  jobNum -1

#Define which runs we want to process (max per submit is 50 jobs!)
try: 
    number = int(sys.argv[1])
except:
    print('number of dataset to process not specified, default is 50')
    number = 50

pax_version = '6.6.5'  # This is only necessary because I name my files by pax_version

# Get the name of the runs for processing
dsets = pd.read_pickle('/home/zhut/data/SingleScatter/data/run_names_v%s_rn.pkl' % pax_version)
run_names = dsets.name.values[:number]

# Use data_process_chain.data_process to check if the set has already been processed
# Print out the ones needed to be submitted
run_submit = []
for run in run_names:
    if data_process(run).check()[0]:
        pass
    elif len(run_submit) % 6 == 5:
        print (" %s " %run, end="")
        print ("%-12s " %data_process(run).check()[1].split('/')[-1][11:])
        run_submit.append(run)
    else:
        print (" %s " %run, end="")
        print ("%-12s " %data_process(run).check()[1].split('/')[-1][11:], end = "")
        run_submit.append(run)  
print ('\n total: %d jobs' %len (run_submit))

# Actual submission
runtmp=0
while (runtmp<len(run_submit)):
    if((working_job() < 50)):
        y = x.format(run=run_submit[runtmp])                    
        #submit_job(y)
        time.sleep(0.1)
        runtmp+=1
        
# Check your jobs with: 'qstat -u <username>'
# Check number of submitted jobs with 'qstat -u <username> | wc -l' (is off by +2 btw)
