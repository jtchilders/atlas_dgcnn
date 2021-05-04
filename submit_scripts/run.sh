#!/bin/bash

#MPIPATH=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/
#export PATH=$MPIPATH/bin:$PATH
#export LD_LIBRARY_PATH=$MPIPATH/lib:$LD_LIBRARY_PATH
CONFIG=configs/atlas_dgcnn_thetagpu.json
cp $CONFIG logdir/$COBALT_JOBID.json
echo $(which mpirun)
mpirun -n $RANKS \
   python main.py -c $CONFIG \
   --logdir $LOGDIR --intraop $OMP_NUM_THREADS --interop $OMP_NUM_THREADS \
   --horovod  #--profiler --batch-term 50