#!/bin/bash

export PATH=/home/chu/mpich-install/bin:$PATH
export PYTHONPATH=/home/chu/tools/boost_1_54_0/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/home/chu/tools/boost_1_54_0/lib:$LD_LIBRARY_PATH
NUMCPUS=$CORES

K=128
LINK=$1
MAXEPOCH=$2
PARTIAL=$3
LANGPAIR=$4
NAME=k${K}.$LANGPAIR.$MAXEPOCH.$PARTIAL.$LINK

nice -15 mpiexec -n $NUMCPUS $PYTHON ./nile.py \
  --f $DATA/train.f \
  --e $DATA/train.e \
  --gold $DATA/train.a.s \
  --ftrees $SOURCE_TREE_DATA/train.f-parse.bin \
  --etrees $TARGET_TREE_DATA/train.e-parse.bin \
  --fdev $DATA/dev.f \
  --edev $DATA/dev.e \
  --ftreesdev $SOURCE_TREE_DATA/dev.f-parse.bin \
  --etreesdev $TARGET_TREE_DATA/dev.e-parse.bin \
  --golddev $DATA/dev.a.s \
  --evcb $DATA/e.vcb \
  --fvcb $DATA/f.vcb \
  --pef $DATA/GIZA++.m4.pef \
  --pfe $DATA/GIZA++.m4.pfe \
  --a1 $DATA/train.m4gdfa.e-f \
  --a2 $DATA/train.nakazawa.e-f.s \
  --a1_dev $DATA/dev.m4gdfa.e-f \
  --a2_dev $DATA/dev.nakazawa.e-f.s \
  --langpair $LANGPAIR \
  --partial $PARTIAL \
  --maxepochs $MAXEPOCH \
  --nto1 $LINK \
  --train \
  --k $K 1> $NAME.out 2> $NAME.err

cat $NAME.out

ITER=`grep F-score-dev $NAME.err | awk '{print $2}' | cat -n | sort -nr -k 2 | head -1 | cut -f 1 | tr -d '[[:space:]]'`
WEIGHTS_FILE=weights.`head -1 $NAME.out`
./weight_extract.py $WEIGHTS_FILE $ITER $NAME
