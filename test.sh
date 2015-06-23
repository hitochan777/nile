#!/bin/bash
#PBS -l walltime=00:30:00,nodes=10:ppn=4
#PBS -N nile-test

# cd $PBS_O_WORKDIR  # Connect to working directory
###################################################################
# Initialize MPI
###################################################################
# export PATH=$HOME/tools/mpich2-install/bin:$PATH
# export PYTHONPATH=/home/nlg-03/riesa/boost_1_48_0/stage/lib:$PYTHONPATH
# export LD_LIBRARY_PATH=/home/nlg-03/riesa/boost_1_48_0/stage/lib:$LD_LIBRARY_PATH
# NUMCPUS=`wc -l $PBS_NODEFILE | awk '{print $1}'`
###################################################################

K=10
DATE=`date +%m%d%y`

BASEDIR=.
DATA=$BASEDIR/data
TEST=$DATA
LANGPAIR=zh-ja
PYTHON=python

WEIGHTS=d062315.k10.n2..weights-2
NAME=$WEIGHTS.test-output.a

mpiexec -n 2 $PYTHON nile.py \
  --f $TEST/dev.f \
  --e $TEST/dev.e \
  --etrees $TEST/dev.e-parse \
  --evcb $TEST/e.vcb \
  --fvcb $TEST/f.vcb \
  --pef $DATA/GIZA++.m4.pef  \
  --pfe $DATA/GIZA++.m4.pfe \
  --score_out score.log \
  --align \
  --langpair $LANGPAIR \
  --weights $WEIGHTS \
  --out $NAME \
  --k $K
