#!/bin/bash

# -------------via chj 2020-12-18 16:57 ---------------------
if [ -e "Ec_out_fcc.dat" ]
then
  rm Ec_out_fcc.dat
fi

# -------------------way1: 单核运算的写法-----------------
export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi

# run lammps simulation
$LAMMPS_BIN  -i in.Zn_fcc> Ec_out_fcc.dat
