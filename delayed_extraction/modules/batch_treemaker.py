# 2016-09, sanderb@nikhef.nl: from Chris with Sander's personalization

# modified by Fei for use

# log files will be created in the folder where you run this script
# For every job you will get an email, make sure this won't be blocked as it will look like spam if you get 500+ emails
# You need write permission for making the processing_dir folder

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

python /home/zhut/data/Delayed/modules/chain_process/data_process_chain.py {run}

rm -rf ${{PROCESSING_DIR}}

"""

# Use submit procedure from CAX
from cax.qsub import submit_job

import os, sys
import time
import pandas as pd
sys.path.append('/home/zhut/data/Delayed/modules/chain_process')
from data_process_chain import data_process


# check my jobs
def working_job():
    cmd="squeue --user=zhut | wc -l"
    jobNum=int(os.popen(cmd).read())
    return  jobNum -1

#Define which runs we want to process (max per submit is 50 jobs!)

pax_version = '6.6.5'  # This is only necessary because I name my files by pax_version
dsets = pd.read_pickle('/home/zhut/data/Delayed/data/run_names_v%s_300.pkl' % pax_version)
# dsets = dsets[dsets.source__type == 'Kr83m']
run_names = dsets.name.values[:]

run_submit = []
for ix, run in enumerate(run_names):
    i = len(run_submit)
    if data_process(run).check()[0]:
        pass
    elif i % 6 == 5:
        print (" %s " %run, end="")
        print (data_process(run).check()[1].split('/')[-1][11:])
        run_submit.append(run)
    else:
        print (" %s " %run, end="")
        run_submit.append(run)
        print (data_process(run).check()[1].split('/')[-1][11:], end = "")
    
print ('\n')
    
runtmp=0
while (runtmp<len(run_submit)):
    if((working_job() < 50)):
        y = x.format(run=run_submit[runtmp])                    
        submit_job(y)
        runtmp+=1
        time.sleep(0.1)

# Check your jobs with: 'qstat -u <username>'
# Check number of submitted jobs with 'qstat -u <username> | wc -l' (is off by +2 btw)
