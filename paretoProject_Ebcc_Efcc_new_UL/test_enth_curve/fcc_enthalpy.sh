#!/bin/bash

# -------------via chj 2020-12-18 16:57 ---------------------
if [ -e "idx_fcc_enthalpy.dat" ]
then
  rm idx_fcc_enthalpy.dat
fi
if [ -e "fcc_result.txt" ]
then
  rm fcc_result.txt
fi

# -------------------way1: 单核运算的写法-----------------
#export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi

# run lammps simulation
#$LAMMPS_BIN  -i in.fcc_enthalpy > fcc_enthalpy.dat

mpirun -np 40 lmp_mpi  -i in.fcc_enthalpy > idx_fcc_enthalpy.dat &
