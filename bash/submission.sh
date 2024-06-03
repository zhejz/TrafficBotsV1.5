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
resume=submission \
action=validate \
trainer.limit_val_batches=1.0 \
datamodule.batch_size_test=3 \
resume.checkpoint=YOUR_WANDB_RUN_NAME:latest \
loggers.wandb.name="validate" \
loggers.wandb.project="TrafficBotsV1.5" \
loggers.wandb.entity="YOUR_ENTITY" \
datamodule.data_dir=${TMPDIR}/datasets \
datamodule.val_scenarios_dir=null \
hydra.run.dir='/cluster/scratch/zhejzhan/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

# ! for validation, set
# action=validate \
# loggers.wandb.name="validate" \

# ! for testing, set
# action=test \
# loggers.wandb.name="test" \


source /cluster/project/cvl/zhejzhan/apps/miniconda3/etc/profile.d/conda.sh
conda activate traffic_bots

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`

echo START copying data: `date`
mkdir $TMPDIR/datasets
cp /cluster/scratch/zhejzhan/$DATASET_DIR/testing.h5 $TMPDIR/datasets/
cp /cluster/scratch/zhejzhan/$DATASET_DIR/validation.h5 $TMPDIR/datasets/
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
