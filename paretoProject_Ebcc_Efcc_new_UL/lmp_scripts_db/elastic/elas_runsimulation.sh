#!/bin/bash
# This is an example bash runscript to run simulations within this
# directory.

#THIS_HOSTNAME=$(hostname -f)
#if [ "$THIS_HOSTNAME" = minerva ]; then
   # settings for Eugene's laptop
 #  export LAMMPS_BIN=/usr/local/bin/lammps

#elif ["$THIS_HOSTNAME" = node01 ]; then
   # settings for hjchen's laptop
 #  export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi
#fi

# -------------via chj 2020-10-14 21:20 ---------------------
if [ -e "elas_out.dat" ]
then
  rm elas_out.dat
fi

#-----------单核的计算方法
export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi

# run lammps simulation
$LAMMPS_BIN  -i in.elastic> elas_out.dat

# 自定义6核并行计算 ——via hjchen  2021-1-5
#mpirun -np 6 lmp_mpi -i in.elastic> elas_out.dat
