#!/bin/bash

MCONDA=/gpfs/jlse-fs0/projects/datascience/parton/conda/2020-12/
source $MCONDA/setup.sh

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=$(nvidia-smi -L | wc -l)
RANKS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS

EXEC=$(which python)
if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
   EXEC="mpirun -n $RANKS -npernode $PPN -hostfile $COBALT_NODEFILE $(which python)"
fi


export OMP_NUM_THREADS=64
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo PATH=$PATH
echo which python = $(which python)
LOGDIR=logdir/$COBALT_JOBID/$(date +"%Y-%m-%d-%H-%M")/conda
mkdir -p $LOGDIR
echo $LOGDIR
cp $0 $LOGDIR/

$EXEC main.py -c configs/atlas_dgcnn_jlse.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
   --logdir $LOGDIR $HOROVOD --profiler --batch-term 50

