
# 基于线程的并行《Python并行计算编程手册》
"""
目前，在软件应用中，使用最为广泛的并发管理编程范式是基于多线程的。
一般来说，应用是由单个进程所启动的，这个进程又会被划分为多个独立的线程，
这些线程表示不同类型的活动，们并行运行，同时又彼此竞争。

多线程编程更倾向于使用共享信息空间来实现线程之间的通信。这种选择使得多线程编程的主要问题变成了对该空间的管理。

# -----------version0----------
基于多线程的并行计算, 但不是多核的mpirun版本
"""
import os
import subprocess
import threading

def function(sh_name):
    cur_dir = os.getcwd()
    print('sh_name = ', sh_name)
    subprocess.run('bash ' + sh_name, shell=True, cwd=cur_dir)
    

if __name__ == '__main__':
    sh_name_list = ['Zn_hcp.sh', 'Zn_fcc.sh', 'Zn_bcc.sh']
 
    t0 = threading.Thread(target=function, args=(sh_name_list[0],))
    t1 = threading.Thread(target=function, args=(sh_name_list[1],))
    t2 = threading.Thread(target=function, args=(sh_name_list[2],))

    t0.start()
    t1.start()
    t2.start()
    
    t0.join()

    t1.join()

    t2.join()
