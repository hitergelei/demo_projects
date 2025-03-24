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

# -------------via chj 2020-12-18 16:57 ---------------------
if [ -e "Ec_out_hcp.dat" ]
then
  rm Ec_out_hcp.dat
elif [ -e "Ec_out_fcc.dat" ]
then
  rm Ec_out_fcc.dat
elif [ -e "Ec_out_bcc.dat" ]
then
  rm Ec_out_bcc.dat

fi

# -------------------way1: 单核运算的写法-----------------
#export LAMMPS_BIN=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi
#
## run lammps simulation
#$LAMMPS_BIN  -i in.Zn_hcp> Ec_out_hcp.dat
#$LAMMPS_BIN  -i in.Zn_fcc> Ec_out_fcc.dat
#$LAMMPS_BIN  -i in.Zn_bcc> Ec_out_bcc.dat


# 自定义3个任务，且每个任务6核并行计算 ——via hjchen  2021-1-5
mpirun -np 6 lmp_mpi -i in.Zn_hcp> Ec_out_hcp.dat & mpirun -np 6 lmp_mpi -i in.Zn_fcc> Ec_out_fcc.dat & mpirun -np 6 lmp_mpi -i in.Zn_bcc> Ec_out_bcc.dat &

# -------------------way2: 多核运算的写法 by hjchen 2020-12-18-------------------

##PBS -N hjchen_Ec_lammps
##PBS -l nodes=1:ppn=40
##PBS -l walltime=48:00:00
##PBS -q batch
##cd $PBS_O_WORKDIR
##source /opt/intel/parallel_studio_xe_2015/psxevars.sh
##NPROCS=`wc -l < $PBS_NODEFILE`      # 运行会报错TODO：Ec_runsimulation.sh:行34: $PBS_NODEFILE: 模糊的重定向
##export lmp_mpi=/home/hjchen/software/lammps-3Mar20/src/lmp_mpi
##mpirun -np $NPROCS -machinefile $PBS_NODEFILE $lmp_mpi -i in.Zn_hcp> Ec_out.dat

