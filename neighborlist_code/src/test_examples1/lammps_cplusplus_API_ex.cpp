#include "lammps.h"

#include <mpi.h>
#include <iostream>


// 怎么编译和运行？https://docs.lammps.org/Build_link.html

/*
conda deactivate
conda activate neighlist_env

export PATH=/home/centos/hjchen/mpich-4.1/_build/bin:$PATH  
export CPATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$CPATH  
export LD_LIBRARY_PATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$LD_LIBRARY_PATH   # 用于运行时的环境
# 或者下面的也行
#export LD_LIBRARY_PATH=/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps:$LD_LIBRARY_PATH

# --->然后编译时执行命令
mpicxx lammps_cplusplus_API_ex.cpp  -o lmp_cpp_api -L/home/centos/hjchen/software/lammps-29Aug2024/src  -llammps

# 或者这样也行
# mpicxx lammps_cplusplus_API_ex.cpp  -o lmp_cpp_api -L/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps  -llammps

*/

int main(int argc, char **argv)
{
    LAMMPS_NS::LAMMPS *lmp;
    // custom argument vector for LAMMPS library
    const char *lmpargv[] = { "liblammps", "-log", "none"};    
    int lmpargc = sizeof(lmpargv)/sizeof(const char *);

    // explicitly initialize MPI
    MPI_Init(&argc, &argv);

    // create LAMMPS instance
    lmp = new LAMMPS_NS::LAMMPS(lmpargc, (char **)lmpargv, MPI_COMM_WORLD);
    // output numerical version string
    std::cout << "LAMMPS version ID: " << lmp->num_ver << std::endl;
    // delete LAMMPS instance
    delete lmp;

    // stop MPI environment
    MPI_Finalize();
    return 0;
}