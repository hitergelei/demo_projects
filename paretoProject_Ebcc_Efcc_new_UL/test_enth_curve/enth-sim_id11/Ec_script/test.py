import os
import subprocess
import threading

def function():
    cur_dir = os.getcwd()
    subprocess.run('bash ' + 'Ec_mpirun.sh', shell=True, cwd=cur_dir)

import time 
start = time.time()
print('-------------1')
os.system('mpirun -np 6 lmp_mpi -i in.Zn_hcp> Ec_out_hcp.dat & mpirun -np 6 lmp_mpi -i in.Zn_fcc> Ec_out_fcc.dat & mpirun -np 6 lmp_mpi -i in.Zn_bcc> Ec_out_bcc.dat ')
end = time.time()
print('------用时{}s'.format(end - start))

#if __name__ == '__main__':
    #function()

