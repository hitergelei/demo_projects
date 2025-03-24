import os
import subprocess


def lmp_run(cur_dir):
     # 例如"xxx/structure_db/E_van_f
    subprocess.run('bash ' + 'Zn_Evf.sh', shell=True, cwd=cur_dir)
    print('The vacancy formation energy calclation is finished!')

if __name__ == '__main__':
    cur = os.getcwd()
    lmp_run(cur)