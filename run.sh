#!/usr/bin/env zsh

CPU=`nproc`
_CORES=`echo "$CPU * 0.8" | bc`

################### CUSTOMIZABLE #####################
LINK=2
ITER=100
PARTIAL=-1
LANG="ja_zh"
export DATA=/windroot/otsuki/data/ASPEC-JC # Absolute path to the base directory which has data
export TARGET_TREE_DATA=$DATA/constituent/1best_unpacked
export SOURCE_TREE_DATA=$DATA/constituent/1best_unpacked
######################################################

export CORES=${_CORES%.*}
export PYTHONPATH=/usr/local/lib:$HOME/developer/pyglog:$HOME/developer/forest_aligner/pyglog:PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/hitoshi/developer/boost_1_59_0:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/hitoshi/developer/boost_1_59_0:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/hitoshi/developer/boost_1_59_0/stage/lib:$LD_LIBRARY_PATH

rm -rf weights-*
rm -rf weights.*
rm -rf k*
rm -rf *output*

set -e

./train.sh $LINK $ITER $PARTIAL $LANG
./test.sh $LINK $ITER $PARTIAL $LANG
