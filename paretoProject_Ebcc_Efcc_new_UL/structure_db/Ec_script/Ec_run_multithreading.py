# 《Python并行计算编程手册》之“基于线程的并行”
"""
目前，在软件应用中，使用最为广泛的并发管理编程范式是基于多线程的。
一般来说，应用是由单个进程所启动的，这个进程又会被划分为多个独立的线程，
这些线程表示不同类型的活动，们并行运行，同时又彼此竞争。

多线程编程更倾向于使用共享信息空间来实现线程之间的通信。这种选择使得多线程编程的主要问题变成了对该空间的管理。
"""
import subprocess
import os
import threading
import numpy as np
import matplotlib.pyplot as plt

"""
version1.多线程的并行计算版本
针对Ec_run.py文件中lmp_run方法中的subprocess.run()方法只能按顺序运行.sh脚本问题，升级成多线程的并行计算版本
"""

# father_path = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), "..")))
# print(father_path)

class Ec_simulation:
    def __init__(self):
        self.idx_list_hcp = []
        self.Ec_list_hcp = []
        self.idx_min_hcp = None   # hcp的晶格常数
        self.Ec_min_hcp = None    # hcp的结合能


        self.idx_list_fcc = []
        self.Ec_list_fcc = []
        self.idx_min_fcc = None   # fcc的晶格常数
        self.Ec_min_fcc = None    # fcc的结合能

        self.idx_list_bcc = []
        self.Ec_list_bcc = []
        self.idx_min_bcc = None   # bcc的晶格常数
        self.Ec_min_bcc = None    # bcc的结合能



    def shell_task(self, sh_name):
        cur_dir = os.getcwd()
        print('sh_name = ', sh_name)
        subprocess.run('bash ' + sh_name, shell=True, cwd=cur_dir)

    def  lmp_run(self):
        sh_name_list = ['Zn_hcp.sh', 'Zn_fcc.sh', 'Zn_bcc.sh']
        # 实例化3个线程（基于多线程的并行计算，缺点：无法满足）
        t0 = threading.Thread(target=self.shell_task, args=(sh_name_list[0],))
        t1 = threading.Thread(target=self.shell_task, args=(sh_name_list[1],))
        t2 = threading.Thread(target=self.shell_task, args=(sh_name_list[2],))
        # 启动线程
        t0.start()
        t1.start()
        t2.start()
        # join()方法会导致调用线程等待，直到它执行完毕
        t0.join()
        t1.join()
        t2.join()
        print('lammps计算结构能量差并行计算结束！')




    def get_value(self):
        # （2）当执行完（1）的lammps计算后，读取Ec_out.dat文件，提取出结合能Ec_min
        # TODO: 考虑用多线程进行这三个任务的计算（写三个方法）
        out_file1 = open('Ec_out_hcp.dat','r')
        for line in out_file1:
            if line.startswith("@@@@"):
                idx_hcp = eval(line.split(":")[1].strip().split(" ")[0])
                Ec_hcp = eval(line.split(":")[1].strip().split(" ")[1])
                if Ec_hcp < 0:   # 只装入<0的数字和索引
                    self.idx_list_hcp.append(idx_hcp)
                    self.Ec_list_hcp.append(Ec_hcp)

        self.idx_min_hcp = self.idx_list_hcp[self.Ec_list_hcp.index(min(self.Ec_list_hcp))]  # 晶格常数
        self.Ec_min_hcp = min(self.Ec_list_hcp)   # 最小的值为结合能
        out_file1.close()


        out_file2 = open('Ec_out_fcc.dat', 'r')
        for line in out_file2:
            if line.startswith("@@@@"):
                idx_fcc = eval(line.split(":")[1].strip().split(" ")[0])
                Ec_fcc = eval(line.split(":")[1].strip().split(" ")[1])
                if Ec_fcc < 0:  # 只装入<0的数字和索引
                    self.idx_list_fcc.append(idx_fcc)
                    self.Ec_list_fcc.append(Ec_fcc)

        self.idx_min_fcc = self.idx_list_fcc[self.Ec_list_fcc.index(min(self.Ec_list_fcc))]  # 晶格常数
        self.Ec_min_fcc = min(self.Ec_list_fcc)  # 最小的值为结合能
        out_file2.close()

        out_file3 = open('Ec_out_bcc.dat', 'r')
        for line in out_file3:
            if line.startswith("@@@@"):
                idx_bcc = eval(line.split(":")[1].strip().split(" ")[0])
                Ec_bcc = eval(line.split(":")[1].strip().split(" ")[1])
                if Ec_bcc < 0:  # 只装入<0的数字和索引
                    self.idx_list_bcc.append(idx_bcc)
                    self.Ec_list_bcc.append(Ec_bcc)

        self.idx_min_bcc = self.idx_list_bcc[self.Ec_list_bcc.index(min(self.Ec_list_bcc))]  # 晶格常数
        self.Ec_min_bcc = min(self.Ec_list_bcc)  # 最小的值为结合能
        out_file3.close()

        deta_hcp_bcc = abs(self.Ec_min_hcp) - abs(self.Ec_min_bcc)   # hcp->bcc的结构能量差
        deta_hcp_fcc = abs(self.Ec_min_hcp) - abs(self.Ec_min_fcc)   # hcp->fcc的结构能量差

        return deta_hcp_bcc, deta_hcp_fcc


# plt.plot(idx_list, Ec_list)
# plt.xlabel('晶格常数/ $a$')
# plt.ylabel('结合能/ $Ec$')
# # plt.xlim(min(idx_list), max(idx_list))
# plt.title('结合能Ec与晶格常数a的关系曲线')
# plt.text(idx_min,Ec_min,(idx_min,Ec_min),color='r')
#
# plt.show()

if __name__ == '__main__':
    import time
    start = time.time()
    Ec_sim = Ec_simulation()
    Ec_sim.lmp_run()

    end = time.time()
    print('lammps cost: %.4f s' %(end-start))
    print(Ec_sim.get_value())
