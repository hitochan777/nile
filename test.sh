#!/bin/zsh
#PBS -l walltime=00:30:00,nodes=10:ppn=4
#PBS -N nile-test

# cd $PBS_O_WORKDIR  # Connect to working directory
###################################################################
# Initialize MPI
###################################################################
export PATH=/home/chu/mpich-install/bin:$PATH
export PYTHONPATH=/home/chu/tools/boost_1_54_0/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/home/chu/tools/boost_1_54_0/lib:$LD_LIBRARY_PATH
NUMCPUS=15
###################################################################

NILE_DIR=/home/otsuki/tools/nile
K=128
LINK=2
MAXEPOCH=100
PARTIAL=-1
BASEDIR=/windroot/chu/nile
DATA=$BASEDIR/data
LANGPAIR=ja_zh
H=65
WEIGHTS=k${K}.$LANGPAIR.$MAXEPOCH.$PARTIAL.$LINK.weights-$H
NAME=$WEIGHTS.test-output.a

echo "nice -19 mpiexec -n $NUMCPUS $PYTHON $NILE_DIR/nile.py \ "
echo " --f $DATA/test.f \ "
echo " --e $DATA/test.e \ "
echo " --ftrees $DATA/test.f-parse \ "
echo " --etrees $DATA/test.e-parse \ "
echo " --evcb $DATA/test.e.vcb \ "
echo " --fvcb $DATA/test.f.vcb \ "
echo " --pef $DATA/GIZA++.m4.pef  \ "
echo " --pfe $DATA/GIZA++.m4.pfe \ "
echo " --a1 $DATA/test.m4gdfa.e-f \ "
echo " --align \ "
echo " --langpair $LANGPAIR \ "
echo " --weights $WEIGHTS \ "
echo "--partial $PARTIAL \ "
echo "--nto1 $LINK \ "
echo " --out $NAME \ "
echo " --k $K "

nice -19 mpiexec -n $NUMCPUS $PYTHON $NILE_DIR/nile.py \
  --f $DATA/test.f \
  --e $DATA/test.e \
  --ftrees $DATA/test.f-parse \
  --etrees $DATA/test.e-parse \
  --evcb $DATA/test.e.vcb \
  --fvcb $DATA/test.f.vcb \
  --pef $DATA/GIZA++.m4.pef  \
  --pfe $DATA/GIZA++.m4.pfe \
  --a1 $DATA/test.m4gdfa.e-f \
  --align \
  --langpair $LANGPAIR \
  --weights $WEIGHTS \
  --partial $PARTIAL \
  --nto1 $LINK \
  --out $NAME \
  --k $K

echo "$NILE_DIR/Fmeasure.py $NAME $DATA/test.a.s"
$NILE_DIR/Fmeasure.py $NAME $DATA/test.a.s

# 3 case
# F-score: 0.76923
# Precision: 0.81266
# Recall: 0.73021
# Correct: 2260
# Hyp Total: 2781
# Gold Total: 3095

# 2 case
# F-score: 0.78040
# Precision: 0.82760
# Recall: 0.73829
# Correct: 2285
# Hyp Total: 2761
# Gold Total: 3095
