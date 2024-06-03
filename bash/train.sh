#!/bin/bash
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.out
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=250G
#SBATCH --open-mode=truncate

trap "echo sigterm recieved, exiting!" SIGTERM

DATASET_DIR="h5_wosac" 
run () {
srun python -u src/run.py \
model=sim_agent \
loggers.wandb.name="train" \
loggers.wandb.project="TrafficBotsV1.5" \
loggers.wandb.entity="YOUR_ENTITY" \
datamodule.data_dir=${TMPDIR}/datasets \
datamodule.val_scenarios_dir=${TMPDIR}/datasets/val_scenarios \
hydra.run.dir='/cluster/scratch/zhejzhan/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

# ! To resume a previous run
# resume.checkpoint=YOUR_WANDB_RUN_NAME:latest \

source /cluster/project/cvl/zhejzhan/apps/miniconda3/etc/profile.d/conda.sh
conda activate traffic_bots

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`

echo START copying data: `date`
mkdir $TMPDIR/datasets
cp /cluster/scratch/zhejzhan/$DATASET_DIR/training.h5 $TMPDIR/datasets/
cp /cluster/scratch/zhejzhan/$DATASET_DIR/validation.h5 $TMPDIR/datasets/
cp -r /cluster/scratch/zhejzhan/$DATASET_DIR/val_scenarios $TMPDIR/datasets/
echo DONE copying: `date`

type run
echo START: `date`
run &
wait
echo DONE: `date`

# time=`date +%Y-%m-%d`
# ./logs/${time}
mkdir -p ./logs/slurm
mv ./logs/$SLURM_JOB_ID.out ./logs/slurm/$SLURM_JOB_ID.out

echo finished at: `date`
exit 0;
