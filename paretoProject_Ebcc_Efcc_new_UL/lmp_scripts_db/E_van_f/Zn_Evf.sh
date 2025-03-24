#!/bin/bash

# -------------via chj 2020-12-18 16:57 ---------------------
if [ -e "Evf.dat" ]
then
  rm Evf.dat
fi

# -------------------way1: 单核计算-----------------
export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi

# run lammps simulation
$LAMMPS_BIN  -i in.Ef_vacancy> Evf.dat

# 自定义6核并行计算 ——via hjchen  2021-1-5

#$mpirun -np 6 lmp_mpi -i in.Ef_vacancy > Evf.dat