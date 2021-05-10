#!/bin/bash

MPIPATH=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/
export PATH=$MPIPATH/bin:$PATH
export LD_LIBRARY_PATH=$MPIPATH/lib:$LD_LIBRARY_PATH

mpirun -n $RANKS -npernode $PPN -hostfile $COBALT_NODEFILE \
   python main.py -c configs/atlas_dgcnn_thetagpu.json \
   --logdir $LOGDIR --intraop $OMP_NUM_THREADS --interop $OMP_NUM_THREADS \
   --horovod --train-more $1 #--profiler --batch-term 50