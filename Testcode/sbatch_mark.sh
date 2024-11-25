#!/bin/bash
#SBATCH -J EC-mark
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p skylake
#! ############################################################
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment
module load cuda/10.2 intel/mkl/2017.4
source ~/mywork/envs/pytorch-env/bin/activate
export OMP_NUM_THREADS=1

##
JOBID=$SLURM_JOB_ID
LOG=models/$model_name/logs/mark-log.$JOBID
ERR=models/$model_name/logs/mark-err.$JOBID

mkdir models/$model_name/logs


echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

python mark.py --file=$filename >> $LOG 2> $ERR

echo "Time: `date`" >> $LOG
