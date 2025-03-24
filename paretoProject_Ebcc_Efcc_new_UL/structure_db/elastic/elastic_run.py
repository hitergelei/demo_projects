import os
import subprocess
import time

class Elastic_simulation:
    def __init__(self):
        self.C11 = None
        self.C12 = None
        self.C13 = None
        self.C33 = None
        self.C44 = None
        self.B = None

    def lmp_run(self):
        cur_dir = os.getcwd()
        subprocess.run('bash ' + 'elas_runsimulation.sh', shell=True, cwd=cur_dir)  # 需要3min左右时间运行完.sh的lammps计算

    def get_value(self):
        """
        读取'elas_out.dat'文件，得到6个目标量qoi
        Returns:
        """
        out_file = open('elas_out.dat','r')  # elas_out.dat以后也可以写成更加通用化的，例如读取.sh文件最后一行，提取出字符串elas_out.dat
        for line in out_file:
            # 下面的if判断用于提取C11,...,B的6个qoi目标量
            if line.startswith("Elastic Constant C11all"):
                self.C11 = eval(line.split(" ")[-2])
            elif line.startswith("Elastic Constant C12all"):
                self.C12 = eval(line.split(" ")[-2])
            elif line.startswith("Elastic Constant C13all"):
                self.C13 = eval(line.split(" ")[-2])
            elif line.startswith("Elastic Constant C33all"):
                self.C33 = eval(line.split(" ")[-2])
            elif line.startswith("Elastic Constant C44all"):
                self.C44 = eval(line.split(" ")[-2])
            elif line.startswith("Bulk Modulus"):
                self.B = eval(line.split(" ")[-2])

            else:   # 以后可以在这里做一个计算elastic任务的时间提取
                pass
        out_file.close()

        return self.C11, self.C12, self.C13, self.C33, self.C44, self.B



if __name__ == '__main__':
    elastic_sim = Elastic_simulation()
    elastic_sim.lmp_run()
    c11, c12, c13, c33, c44, b = elastic_sim.get_value()
    print('--------弹性常数计算结果>>>')
    print('B={}, C11={}, C12={}, C44={}, C33={}, C13={}'.format(b, c11, c12, c44, c33, c13))
    print()
    