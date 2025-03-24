import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

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



    def  lmp_run(self):
        # （1）bash运行Ec的shell脚本文件，生成结果文件Ec_out.dat
        cur_dir = os.getcwd()
        # print('--------------------: ', cur_dir)
        #subprocess.run('bash ' + 'Ec_mpirun.sh', shell=True, cwd=cur_dir)

        # subprocess.Popen('bash ' + 'Zn_hcp.sh', shell=True, cwd=cur_dir)
        # subprocess.Popen('bash ' + 'Zn_fcc.sh', shell=True, cwd=cur_dir)
        # subprocess.Popen('bash ' + 'Zn_bcc.sh', shell=True, cwd=cur_dir)
        #
        subprocess.run('bash ' + 'Zn_hcp.sh', shell=True, cwd=cur_dir)
        subprocess.run('bash ' + 'Zn_fcc.sh', shell=True, cwd=cur_dir)
        subprocess.run('bash ' + 'Zn_bcc.sh', shell=True, cwd=cur_dir)
        print('lammps计算结构能量差结束！')




    def get_value(self):
        # （2）当执行完（1）的lammps计算后，读取Ec_out.dat文件，提取出结合能Ec_min
        # TODO: 考虑用多线程进行这三个任务的计算（写三个方法）
        out_file1 = open('Ec_out_hcp.dat','r')
        for line in out_file1.readlines():
            if line.startswith("@@@@"):
                idx_hcp = eval(line.split(":")[1].strip().split(" ")[0])
                Ec_hcp = eval(line.split(":")[1].strip().split(" ")[1])
                #if Ec_hcp < 0:   # 只装入<0的数字和索引
                self.idx_list_hcp.append(idx_hcp)
                self.Ec_list_hcp.append(Ec_hcp)

        self.idx_min_hcp = self.idx_list_hcp[self.Ec_list_hcp.index(min(self.Ec_list_hcp))]  # 晶格常数
        self.Ec_min_hcp = min(self.Ec_list_hcp)   # 最小的值为结合能
        out_file1.close()


        out_file2 = open('Ec_out_fcc.dat', 'r')
        for line in out_file2.readlines():
            if line.startswith("@@@@"):
                idx_fcc = eval(line.split(":")[1].strip().split(" ")[0])
                Ec_fcc = eval(line.split(":")[1].strip().split(" ")[1])
                #if Ec_fcc < 0:  # 只装入<0的数字和索引
                self.idx_list_fcc.append(idx_fcc)
                self.Ec_list_fcc.append(Ec_fcc)

        self.idx_min_fcc = self.idx_list_fcc[self.Ec_list_fcc.index(min(self.Ec_list_fcc))]  # 晶格常数
        self.Ec_min_fcc = min(self.Ec_list_fcc)  # 最小的值为结合能
        out_file2.close()

        out_file3 = open('Ec_out_bcc.dat', 'r')
        for line in out_file3.readlines():
            if line.startswith("@@@@"):
                idx_bcc = eval(line.split(":")[1].strip().split(" ")[0])
                Ec_bcc = eval(line.split(":")[1].strip().split(" ")[1])
                #if Ec_bcc < 0:  # 只装入<0的数字和索引
                self.idx_list_bcc.append(idx_bcc)
                self.Ec_list_bcc.append(Ec_bcc)

        self.idx_min_bcc = self.idx_list_bcc[self.Ec_list_bcc.index(min(self.Ec_list_bcc))]  # 晶格常数
        self.Ec_min_bcc = min(self.Ec_list_bcc)  # 最小的值为结合能
        out_file3.close()

        deta_hcp_bcc = abs(self.Ec_min_hcp) - abs(self.Ec_min_bcc)   # hcp->bcc的结构能量差
        deta_hcp_fcc = abs(self.Ec_min_hcp) - abs(self.Ec_min_fcc)   # hcp->fcc的结构能量差

        return self.idx_min_hcp, self.Ec_min_hcp, self.idx_min_bcc, self.Ec_min_bcc, self.idx_min_fcc, self.Ec_min_fcc, deta_hcp_bcc, deta_hcp_fcc


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
    #Ec_sim.lmp_run()

    end = time.time()
    print('lammps cost: %.4f s' %(end-start))
    print('a(hcp), Ec(hcp), a(bcc), Ec(bcc), a(fcc), Ec(fcc), hcp-bcc, hcp-fcc = \n', Ec_sim.get_value())

