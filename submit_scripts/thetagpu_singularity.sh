#!/bin/bash
#COBALT -O logdir/$COBALT_JOBID
#COBALT -t 360
#COBALT -n 8
#COBALT -A datascience

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)

# get home directory without symobolic links
FULL_HOME=$( cd $HOME && pwd -LP)
echo FULL_HOME=$FULL_HOME

# MPIPATH=/usr/local/mpi
# MPIPATH=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/
# echo MPIPATH=$MPIPATH

echo COBALT_NODEFILE=$COBALT_NODEFILE
echo COBALT_JOBID=$COBALT_JOBID

# export LD_LIBRARY_PATH=$MPIPATH/lib:$LD_LIBRARY_PATH
# export PATH=$MPIPATH/bin:$PATH
# echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# echo PATH=$PATH

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8
RANKS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS
echo NODES=$(cat $COBALT_NODEFILE)

#export SINGULARITYENV_LD_LIBRARY_PATH=$MPIPATH/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs

export OMP_NUM_THREADS=8
CONTAINER=/lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.12-py3.simg
echo CONTAINER=$CONTAINER
export LOGDIR=logdir/$COBALT_JOBID/$(date +"%Y-%m-%d-%H-%M")
mkdir -p $LOGDIR
echo LOGDIR=$LOGDIR
cp $0 $LOGDIR/$COBALT_JOBID.sh
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
#export TF_XLA_FLAGS=--tf_xla_auto_jit=1
#export TF_XLA_FLAGS=--tf_xla_auto_jit=fusible
singularity exec --nv -B /lus/theta-fs0/software/thetagpu -B $FULL_HOME \
   -B /lus/theta-fs0/projects/datascience/parton/ \
   -B /lus/theta-fs0/projects/atlasMLbjets/parton \
   -B /lus/theta-fs0/software/datascience/ $CONTAINER \
      submit_scripts/run.sh
