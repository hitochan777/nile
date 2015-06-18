#!/bin/bash
#PBS -l walltime=00:30:00,nodes=10:ppn=4
#PBS -N nile-train

# cd $PBS_O_WORKDIR  # Connect to working directory
###################################################################
# Initialize MPI
###################################################################
# export PATH=/home/nlg-03/riesa/mpich2-install/bin:$PATH
# export PYTHONPATH=/home/nlg-03/riesa/boost_1_48_0/stage/lib:$PYTHONPATH
# export LD_LIBRARY_PATH=/home/nlg-03/riesa/boost_1_48_0/stage/lib:$LD_LIBRARY_PATH
# NUMCPUS=`wc -l $PBS_NODEFILE | awk '{print $1}'`
###################################################################

NUMCPUS=2

K=10
DATE=`date +%m%d%y`

BASEDIR=.
DATA=$BASEDIR/data
TRAIN=$DATA
DEV=$DATA
PYTHON=python

NAME=d$DATE.k${K}.n$NUMCPUS.$LANGPAIR
echo "mpiexec -n $NUMCPUS $PYTHON nile.py\\"
echo "--f $TRAIN/train.f\\"
echo "--e $TRAIN/train.e\\"
echo "--gold $TRAIN/train.a\\"
echo "--etrees $TRAIN/train.e-parse\\"
# echo "--ftrees $TRAIN/train.f-parse\\"
echo "--fdev $DEV/dev.f\\"
echo "--edev $DEV/dev.e\\"
echo "--etreesdev $DEV/dev.e-parse\\"
# echo "--ftreesdev $DEV/dev.f-parse\\"
echo "--golddev $DEV/dev.a\\"
echo "--evcb $DATA/e.vcb\\"
echo "--fvcb $DATA/f.vcb\\"
echo "--pef $DATA/GIZA++.m4.pef\\"
echo "--pfe $DATA/GIZA++.m4.pfe\\"
echo "--langpair zh_ja\\"
echo "--train\\"
echo "--k $K 1> $NAME.out 2> $NAME.err\\"

nice -15 mpiexec -n $NUMCPUS $PYTHON nile.py \
  --f $TRAIN/train.f \
  --e $TRAIN/train.e \
  --gold $TRAIN/train.a \
  --etrees $TRAIN/train.e-parse \
  --fdev $DEV/dev.f \
  --edev $DEV/dev.e \
  --etreesdev $DEV/dev.e-parse \
  --golddev $DEV/dev.a \
  --evcb $DATA/e.vcb \
  --fvcb $DATA/f.vcb \
  --pef $DATA/GIZA++.m4.pef  \
  --pfe $DATA/GIZA++.m4.pfe \
  --langpair zh_ja \
  --maxepochs 2 \
  --train \
  --k $K 1> $NAME.out 2> $NAME.err
