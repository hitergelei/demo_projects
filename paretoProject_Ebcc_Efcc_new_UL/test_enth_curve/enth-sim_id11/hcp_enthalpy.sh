#!/bin/bash

# -------------via chj 2020-12-18 16:57 ---------------------
if [ -e "idx_hcp_enthalpy.dat" ]
then
  rm idx_hcp_enthalpy.dat
fi
if [ -e "hcp_result.txt" ]
then
  rm hcp_result.txt
fi
# -------------------way1: 单核运算的写法-----------------
#export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi

# run lammps simulation
mpirun -np 40 lmp_mpi  -i in.hcp_enthalpy > idx_hcp_enthalpy.dat &

