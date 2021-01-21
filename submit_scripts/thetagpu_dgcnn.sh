#!/bin/bash -l
#COBALT -O logdir/$COBALT_JOBID
#COBALT -t 360
#COBALT -n 8
#COBALT -A datascience

#MCONDA=/lus/theta-fs0/software/thetagpu/conda/tf_master/2020-12-17/mconda3
#MCONDA=/lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3
#MCONDA=/lus/theta-fs0/software/thetagpu/conda/tf_master/2020-12-23/mconda3
MCONDA=/lus/theta-fs0/software/thetagpu/conda/tf_master/2021-01-08/mconda3
source $MCONDA/setup.sh

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8
RANKS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS

EXEC=$(which python)
if [ $RANKS -gt 1 ]; then
   echo [$SECONDS] adding horovod with $RANKS ranks
   HOROVOD=--horovod
   EXEC="mpirun -n $RANKS -npernode $PPN -hostfile $COBALT_NODEFILE $(which python)"
fi

echo EXEC=$EXEC
echo NODES=$(cat $COBALT_NODEFILE)
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
