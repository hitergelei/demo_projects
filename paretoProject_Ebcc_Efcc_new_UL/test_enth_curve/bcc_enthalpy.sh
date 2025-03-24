#!/bin/bash

# -------------via chj 2020-12-18 16:57 ---------------------
if [ -e "idx_bcc_enthalpy.dat" ]
then
  rm idx_bcc_enthalpy.dat
fi

if [ -e "bcc_result.txt" ]
then
  rm bcc_result.txt
fi

# -------------------way1: 单核运算的写法-----------------
#export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi

# run lammps simulation
#$LAMMPS_BIN  -i in.bcc_enthalpy> bcc_enthalpy.dat

mpirun -np 40 lmp_mpi  -i in.bcc_enthalpy > idx_bcc_enthalpy.dat &
