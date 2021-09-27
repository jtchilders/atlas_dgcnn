#!/bin/bash -l
#COBALT -O logdir/$COBALT_JOBID
#COBALT -t 360
#COBALT -n 8
#COBALT -A datascience

# load conda environment
module load conda/2021-09-22
conda activate

# count number of nodes job has allocated to it
NODES=`cat $COBALT_NODEFILE | wc -l`
# ThetaGPU has 8 GPUs per node
PPN=8
# total MPI ranks to run
RANKS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS

# if I have more than 1 MPI rank, use Horovod for parallel training
EXEC=$(which python)
if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
   EXEC="mpirun -n $RANKS -npernode $PPN -hostfile $COBALT_NODEFILE $(which python)"
fi

echo EXEC=$EXEC
echo NODES=$(cat $COBALT_NODEFILE)
# set OMP_NUM_THREADS for CPU-side operations
export OMP_NUM_THREADS=8
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PATH=$PATH
echo which python = $(which python)
LOGDIR=logdir/$COBALT_JOBID/$(date +"%Y-%m-%d-%H-%M")
mkdir -p $LOGDIR
echo $LOGDIR
cp $0 logdir/$COBALT_JOBID.sh
CONFIG=configs/atlas_dgcnn_thetagpu.json
cp $CONFIG logdir/$COBALT_JOBID.json
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
# export TF_XLA_FLAGS=--tf_xla_auto_jit=1
#export TF_XLA_FLAGS=--tf_xla_auto_jit=fusible
$EXEC main.py -c $CONFIG --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
   --logdir $LOGDIR $HOROVOD # --batch-term 20 #--profiler 
