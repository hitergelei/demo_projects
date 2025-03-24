import numpy as np
import scipy.stats
import tool_engine.kde as kde
import os
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from input import potential_param_config as config
import gc
import copy

from input.DFT_qois import element_name, qoi_name_list, qoi_value_list, DFT_qois_dict

from tool_engine import lammps


class PyPosmatError(Exception):
  """Exception handling class for pyposmat"""
  def __init__(self, value):
    self.value = value

  def __str__(self):
    return repr(self.value)



class PyPosmatEngine(object):    # 对应源代码的pyposmat.py文件的class PyPosmatEngine(object)内容
    """
    用于势函数的参数采样
    """
    def __init__(self,  is_restart = False):

        self._is_restart = is_restart  # if set to true,
                                       # try to recover previous simulations



        # configure results file
        self._fname_out = 'pyposmat.out'    # by hjchen 2020-12-11
        self._f_results = open(self._fname_out,'w')

        # configure log file
        self._fname_log = 'pyposmat.log'      # by hjchen 2020-12-11
        self._f_log = open(self._fname_log, 'w')   #  self._f_log.name = 'pyposmat.log'



        #
        self._param_names = list(config.parameter_distribution.keys())   # 势函数的参数名：List类型
        self._param_info = config.parameter_distribution   # 势函数的参数名和初始分布：字典类型
        self._qoi_names = qoi_name_list    # qois的名字：List类型

        self._sampler_type = None

        self._error_names = ["{}.err".format(q) for q in self._qoi_names]  # self._error_names共有10个.err
        self._names = self._param_names + self._qoi_names + self._error_names    #  self._names为.out文件的第一行数据，但注意self._names为31个（即，除掉sim_id）
        self._types = len(self._param_names) * ['param'] \
                + len(self._qoi_names) * ['qoi'] \
                + len(self._error_names) * ['err']   # self._types为.out文件的第二行数据（共有31个）


# ---------------------------------------------~~~~~~~added by 陈洪剑~~~~~~~----------------------------------------------


# +++++++++++++++++++++++++++++++++++++ （0）给定一组势函数参数就自动生成势函数列表文件 +++++++++++++++++++++++++++++++++++
    # 生成势函数列表文件
    def potential_file_gen(self, pparam_dict):  # 例如传入的势函数参数有28个
        """
        每传入一组势函数参数，就对应在xxx/lmp_eam_fs_gen目录下更新势函数列表文件xxx.eam.fs
        Args:
            pparam_dict: 势函数参数

        Returns:None

        """
        # ---------------1).★定义好一些中间变量（函数值）--------------------
        emb = [None for i in range(10000)]  # 嵌入能的值
        phi_eff = [None for i in range(10000)]  # 电子密度函数值
        vsum = [None for i in range(10000)]  # 对势函数值
        v = [None for i in range(10000)]  # 乘以r后的对势值

        # ---------------2).★定义好一些固定输入值--------------------
        # 对结点位置Rk和rk进行赋值（不需要调参的）
        self.Rk = config.Rk_INPUT
        self.rk = config.rk_INPUT

        self.qele1 = config.qele1  # for Zn
        self.qele2 = config.qele2  # for Zn
        # ----------
        self.drho = config.drho  # 即drho
        self.dr = config.dr    # dr
        # --------------------------------------------------------

        # ---------------3).★定义好一些势函数参数字典: 共28个参数--------------------
        ZBL = {}       # ZBL参数：4个
        Guage = {}     # Guage参数：2个
        Ak = {}        # Ak电子密度函数的节点系数参数：10个
        ak = {}        # ak对势函数的节点系数参数：10个
        eam = {}       # 嵌入能的参数A和B：2个

        for key in pparam_dict.keys():
            if key.startswith('ZBL_'):
                ZBL[key.split('ZBL_')[1]] = pparam_dict[key]
            elif key.startswith('Guage_'):
                Guage[key.split('Guage_')[1]] = pparam_dict[key]
            elif key.startswith('den_A'):
                Ak[str(int(key.split('den_A')[1])-1)] = pparam_dict[key]
            elif key.startswith('P_a'):
                ak[str(int(key.split('P_a')[1])-1)] = pparam_dict[key]
            elif key.startswith('eam_'):
                eam[key.split('eam_')[1]] = pparam_dict[key]
            else:
                raise KeyError("输入的%s的类型不是势函数的参数类型！" % key)

        def HH(x):  # Hs(x)是单位阶跃函数
            if x > 0:
                return 1
            else:
                return 0

        # TODO： 这个势函数列表文件需要写入到'lmp_eam_fs_gen'文件的目录下，对应的qoi的.sh脚本同级的potential.mod中eam.fs路径也要修改过来！！！
        root_path = os.getcwd()   # 显示root目录，例如：root_path = '/home/hjchen/tmp/Zn_pymoo_20201214'
        file_name = os.path.join(root_path, 'lmp_eam_fs_gen', '%s_2021chj.eam.fs'%element_name)  # root目录下的lmp_eam_fs_gen下的xxx.eam.fs势函数列表文件
        file = open(file_name, 'w')  # 势函数文件列表 (TODO: 可以考虑拷贝势函数文件+标识符作为每次生成的文件，存放到某个文件夹下面！！）

        file.write("%s potential\n"%element_name)
        file.write("Finnis-Sinclair formalism\n")
        file.write("writen by hjchen in %s\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        file.write("1  %s\n"%element_name)
        file.write("10000  2.0000E-0003  10000  6.3000E-0004  6.3000E+0000\n")
        file.write("12   2.43050E+0001   3.2090E+0000  hcp\n")  # 把晶格常数写死了就是3.209

        # ----------------嵌入能的计算
        for i in range(0, 10000):
            rho = float(i) * self.drho  # 对应ρ = i * dρ
            if i == 0:
                emb[i] = 0.0e0
            else:
                # 进行规范不变性变换后，嵌入能公式F_eff(ρ) = F(ρ/S) + (C/S)*ρ
                # 其中，F(ρi) = -√ρi
                emb[i] = - np.sqrt(rho / Guage['S']) + (Guage['C'] / Guage['S']) * rho

        for i in range(0, int(10000 / 5)):  # 分成5列写入到势函数文件中
            file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
                       format(emb[5 * i], emb[5 * i + 1], emb[5 * i + 2], emb[5 * i + 3],
                              emb[5 * i + 4]))


        # ----------------电子密度函数Φ(r)的计算
        for i in range(0, 10000):
            r = float(i) * self.dr  # 对应r = i * dr
            # fr1对应的是未进行规范不变性变换的对势函数Φ(r)
            phi_r = Ak['0'] * (self.Rk[0] - r) ** 3 * HH(self.Rk[0] - r) + \
                    Ak['1'] * (self.Rk[1] - r) ** 3 * HH(self.Rk[1] - r) + \
                    Ak['2'] * (self.Rk[2] - r) ** 3 * HH(self.Rk[2] - r) + \
                    Ak['3'] * (self.Rk[3] - r) ** 3 * HH(self.Rk[3] - r) + \
                    Ak['4'] * (self.Rk[4] - r) ** 3 * HH(self.Rk[4] - r) + \
                    Ak['5'] * (self.Rk[5] - r) ** 3 * HH(self.Rk[5] - r) + \
                    Ak['6'] * (self.Rk[6] - r) ** 3 * HH(self.Rk[6] - r) + \
                    Ak['7'] * (self.Rk[7] - r) ** 3 * HH(self.Rk[7] - r) + \
                    Ak['8'] * (self.Rk[8] - r) ** 3 * HH(self.Rk[8] - r) + \
                    Ak['9'] * (self.Rk[9] - r) ** 3 * HH(self.Rk[9] - r)
            # fr对应的是规范不变性变换后的对势函数Φ_eff(r)
            phi_eff[i] = Guage['S'] * phi_r  # 对应公式：Φ_eff(r) = S * Φ(r)

        for i in range(0, int(10000 / 5)):  # 分成5列写入到势函数文件中
            file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
                       format(phi_eff[5 * i], phi_eff[5 * i + 1], phi_eff[5 * i + 2], phi_eff[5 * i + 3],
                              phi_eff[5 * i + 4]))

        for i in range(0, 10000):
            r = float(i) * self.dr
            if r == 0:
                r = 0.1e-12
            if i == 0:
                Zi = self.qele1  # 对应Zi
                Zj = self.qele2  # 对应Zj
                ev = 1.602176565e-19  # 对应e
                pi = 3.14159265358979324e0  # 对应π
                epsil0 = 8.854187817e-12  # 对应ε0
                bohrad = 0.52917721067e0  # 波尔半径：0.529埃
                exn = 0.23e0  # 幂函数的指数（常数）
                beta = 1 / (4.0 * pi * epsil0) * (Zi * Zj * np.power(ev, 2)) * 1.0e10 / ev
                a = 0.8854 * bohrad / (np.power(Zi, exn) + np.power(Zj, exn))  # 对应公式a = 0.4685335/(Zi^0.23 + Zj^0.23)
                # bzb[0] = -3.19980 / a     #  对应x = rij/a
                # bzb[1] = -0.94229 / a     #  对应x = rij/a
                # bzb[2] = -0.40290 / a     #  对应x = rij/a
                # bzb[3] = -0.20162 / a     #  对应x = rij/a
                v[i] = beta * (0.18175 * np.exp(-3.19980 * r / a) +
                               0.50986 * np.exp(-0.94229 * r / a) +
                               0.28022 * np.exp(-0.40290 * r / a) +
                               0.02817 * np.exp(-0.20162 * r / a))

            # (1) r < rm时，计算的对势公式是V_ZBL
            if r < 1.0:
                Zi = self.qele1
                Zj = self.qele2
                ev = 1.602176565e-19
                pi = 3.14159265358979324e0
                epsil0 = 8.854187817e-12
                bohrad = 0.52917721067e0
                exn = 0.23e0
                # beta的公式
                beta = 1 / (4.0 * pi * epsil0) * (Zi * Zj * np.power(ev, 2)) * 1.0e10 / ev
                a = 0.8854 * bohrad / (np.power(Zi, exn) + np.power(Zj, exn))
                # bzb[0] = -3.19980/a
                # bzb[1] = -0.94229/a
                # bzb[2] = -0.40290/a
                # bzb[3] = -0.20162/a
                rinv = 1.0 / r
                vsum[i] = beta * rinv * (0.18175 * np.exp(-3.19980 * r / a) +
                                         0.50986 * np.exp(-0.94229 * r / a) +
                                         0.28022 * np.exp(-0.40290 * r / a) +
                                         0.02817 * np.exp(-0.20162 * r / a))

            # (2) rm <= r <= rn时，计算的对势公式是：V_Connect
            elif r >= 1.0 and r < 2.3:
                vsum[i] = np.exp(ZBL['B0'] + ZBL['B1'] * r + ZBL['B2'] * np.power(r, 2) + ZBL['B3'] * np.power(r, 3))

            # (3) r >=rn时，计算的对势公式是：V_Original
            elif r >= 2.3:
                vsum[i] = ak['0'] * (self.rk[0] - r) ** 3 * HH(self.rk[0] - r) + \
                          ak['1'] * (self.rk[1] - r) ** 3 * HH(self.rk[1] - r) + \
                          ak['2'] * (self.rk[2] - r) ** 3 * HH(self.rk[2] - r) + \
                          ak['3'] * (self.rk[3] - r) ** 3 * HH(self.rk[3] - r) + \
                          ak['4'] * (self.rk[4] - r) ** 3 * HH(self.rk[4] - r) + \
                          ak['5'] * (self.rk[5] - r) ** 3 * HH(self.rk[5] - r) + \
                          ak['6'] * (self.rk[6] - r) ** 3 * HH(self.rk[6] - r) + \
                          ak['7'] * (self.rk[7] - r) ** 3 * HH(self.rk[7] - r) + \
                          ak['8'] * (self.rk[8] - r) ** 3 * HH(self.rk[8] - r) + \
                          ak['9'] * (self.rk[9] - r) ** 3 * HH(self.rk[9] - r) - \
                          2.0 * Guage['C'] * (Ak['0'] * (self.Rk[0] - r) ** 3 * HH(self.Rk[0] - r) +
                                              Ak['1'] * (self.Rk[1] - r) ** 3 * HH(self.Rk[1] - r) +
                                              Ak['2'] * (self.Rk[2] - r) ** 3 * HH(self.Rk[2] - r) +
                                              Ak['3'] * (self.Rk[3] - r) ** 3 * HH(self.Rk[3] - r) +
                                              Ak['4'] * (self.Rk[4] - r) ** 3 * HH(self.Rk[4] - r) +
                                              Ak['5'] * (self.Rk[5] - r) ** 3 * HH(self.Rk[5] - r) +
                                              Ak['6'] * (self.Rk[6] - r) ** 3 * HH(self.Rk[6] - r) +
                                              Ak['7'] * (self.Rk[7] - r) ** 3 * HH(self.Rk[7] - r) +
                                              Ak['8'] * (self.Rk[8] - r) ** 3 * HH(self.Rk[8] - r) +
                                              Ak['9'] * (self.Rk[9] - r) ** 3 * HH(self.Rk[9] - r))

            # r_value.append(r)  # 把r存起来
            # pair_potential.append(vsum[i])  # 把对势的值存起来（没有乘以r的）
            v[i] = r * vsum[i]

        for i in range(0, int(10000 / 5)):  # 分成5列写入到势函数文件中
            file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
                       format(v[5 * i], v[5 * i + 1], v[5 * i + 2], v[5 * i + 3],
                              v[5 * i + 4]))

        file.close()




    # +++++++++++++++++++++++++++++++++++++ （1）定义均匀采样的代码 +++++++++++++++++++++++++++++++++++

    def get_uniform_sample(self):    #  第一次的采样方法：参数的均匀分布采样

        # initialize param dict（初始化参数的值为None）
        # self._param_names = list(self._param_info.keys())  # 获得28个参数名称
        param_dict = {}
        for pn in self._param_names:  # self._param_names = ['chrg_Mg', 'chrg_O', 'MgMg_A', 'MgMg_rho', 'MgMg_C', 'MgO_A', 'MgO_rho', 'MgO_C', 'OO_A', 'OO_rho', 'OO_C']
            param_dict[pn] = None  # 赋值得到 param_dict = {'chrg_Mg': None, 'chrg_O': None, 'MgMg_A': None, 'MgMg_rho': None, 'MgMg_C': None, 'MgO_A': None, 'MgO_rho': None, 'MgO_C': None, 'OO_A': None, 'OO_rho': None, 'OO_C': None}

        for pn in self._param_names:
            if self._param_info[pn]['type'] == 'uniform':    #  对6个自由参数进行均匀采样，例如，当pn = 'chrg_Mg'时， self._param_info['chrg_Mg'] = {'type': 'uniform', 'info': [1.5, 2.5]}
                a = self._param_info[pn]['info']['a']   #  当pn = 'chrg_Mg'时, a = 1.5   #  每次迭代时，均匀分布的下界a和b都是固定的，跟配置文件的模糊均匀分布一样？
                b = self._param_info[pn]['info']['b']   #  当pn = 'chrg_Mg'时, b = 2.5
                param_dict[pn] = np.random.uniform(a,b)  # 例如：np.random.uniform(a,b) = 2.4462101297886916，则param_dict = {'chrg_Mg': 2.4462101297886916}
            # elif self._param_info[pn]['type'] == 'static':
            #     param_dict[pn] = self._param_info[pn]['info']['value']
            else:
                raise Exception("未知的势能参数类型！")
        return param_dict  # 更新势能参数的具体取值



# +++++++++++++++++++++++++++++++++++++ （2）定义kde采样的代码 +++++++++++++++++++++++++++++++++++
    # ---------------陈洪剑2020-12-8号修改：增加带宽估计---------------
    def determine_kde_bandwidth(self, X, kde_bw_type):

        if kde_bw_type == 'chiu1999':
            kde_bw = kde.Chiu1999_h(X)
        elif kde_bw_type == 'silverman1986':
            kde_bw = kde.Silverman1986_h(X)
        return kde_bw

    # ---------------陈洪剑2020-12-8号修改：增加带宽估计---------------

    def get_free_parameter_list(self):   # 自由参数的名字获取
        self._free_param_list = []   #  获取是均匀分布的势能参数的参数名列表（也即自由参数的名称）
        for pn in self._param_names:  # self._param_names = ['chrg_Mg','chrg_O','MgMg_A','MgMg_rho','MgMg_C','MgO_A','MgO_rho','MgO_C','OO_A','OO_rho','OO_C']
            if self._param_info[pn]['type'] == 'uniform':    # 例如，当pn = 'chrg_Mg'时， self._param_info['chrg_Mg'] = {'type': 'uniform', 'info': [1.5, 2.5]}
                self._free_param_list.append(pn)   # 有6个自由参数名称

        return self._free_param_list


    def initialize_kde_sampler(self,fname_in):  #  例如，第2次迭代，此时fname__in = 'culled_001.out'；第3次迭代，此时fname_in = 'culled_002.out'，以此类推
        f = open(fname_in,'r')   #  fname_in = 'culled_00x.out'，具体内容形式可以参考\pyflamestk\examples\lmps_MgO_pareto_post\'culled.out'
        lines = f.readlines()
        f.close()

        self._kde_names = lines[0].strip().split(',')   #  获取culled_00x.out'第2行的内容（表头名字）
        self._kde_names = [str(v.strip()) for v in self._kde_names]   # self._kde_names = ['sim_id','chrg_Mg','chrg_O','MgMg_A','MgMg_rho','MgMg_C','MgO_A','MgO_rho','MgO_C','OO_A','OO_rho','OO_C','MgO_NaCl.a0','MgO_NaCl.c11','MgO_NaCl.c12','MgO_NaCl.c44','MgO_NaCl.B','MgO_NaCl.G','MgO_NaCl.fr_a','MgO_NaCl.fr_c','MgO_NaCl.sch','MgO_NaCl.001s','MgO_NaCl.a0.err','MgO_NaCl.c11.err','MgO_NaCl.c12.err','MgO_NaCl.c44.err','MgO_NaCl.B.err','MgO_NaCl.G.err','MgO_NaCl.fr_a.err','MgO_NaCl.fr_c.err','MgO_NaCl.sch.err','MgO_NaCl.001s.err']

        self._kde_types = lines[1].strip().split(',')   #  获取culled_00x.out'第2行的内容
        self._kde_types = [str(v).strip() for v in self._kde_types] # 去掉字符串中的空格。self._kde_types = ['sim_id','param','param','param','param','param','param','param','param','param','param','param','qoi','qoi','qoi','qoi','qoi','qoi','qoi','qoi','qoi','qoi','err','err','err','err','err','err','err','err','err','err']

        datas = []
        for i in range(2,len(lines)):   #  #  获取culled_00x.out'第3行~最后行的数据集
            line = lines[i]
            line = line.strip().split(',')
            line = [float(v) for v in line]
            datas.append(line)
        datas = np.array(datas)

        # construct free parameter list
        free_param_list = self.get_free_parameter_list()  # 得到free_param_list = ['chrg_Mg', 'MgO_A', 'MgO_rho', 'OO_A', 'OO_rho', 'OO_C
        self._kde_free_param_indx = []
        for i,v in enumerate(self._kde_names):  # self._kde_names是32列的
            if v in free_param_list:
                self._kde_free_param_indx.append(i)  # 从32列名字中获得自由参数的索引列：self._kde_free_param_indx = [1, 6, 7, 9, 10, 11]
        # DEBUG
        free_params = datas[:,self._kde_free_param_indx]  #  获得culled_00x.out中的6个自由参数取值对应的全部数据（自由参数的全部取值数据），例如result_00x.out中数据量为10000时，culled_00x.out的shape形状为：(2074, 6)

        # ------------------陈洪剑2020-12-8号修改：增加带宽估计---------------------
        # print('# ------------------陈洪剑2020-12-8号修改：增加带宽估计---------------------')
        print("#--------------kde_kernel增加带宽估计'silverman'方法，仍用强边界条件判断：2021-1-3")
        self._X = free_params.transpose()
        print('>>>>>>> self._X = \n', self._X)
        print('>>>>>>> self._x.shape = ', self._X.shape)
        # kde_bw = self.determine_kde_bandwidth(X=self._X, kde_bw_type="chiu1999")
        # self._kde_kernel = scipy.stats.gaussian_kde(self._X, kde_bw)   #  np.transpose用法：转置。比如(2074, 6)转置后成为(6, 2074),然后进行gaussian_kde估计得到概率密度函数

        # via chj 2021-1-3，由于增加chiu1999带宽估计，会导致后面采样时边界条件无法满足而陷入死循环，
        # way1: 故不增加带宽估计，采用新的边界约束条件（弱化的）：只要自由参数位于初始定义的上下界里，就当做采样成功。
        # self._kde_kernel = scipy.stats.gaussian_kde(self._X)

        # way2: 增加silverman带宽估计，可以通过。强化的边界条件判断：仍基于culled_xxx.out的数据集作为上下界判断。
        self._kde_kernel = scipy.stats.gaussian_kde(self._X, 'silverman')
        # ------------------陈洪剑2020-12-8号修改---------------------


    def get_kde_sample(self):  # 第二次采样时（即第二次迭代时）的方法，触发kde采样机制
        if self._kde_kernel is None:
            self.initialize_kde_sampler(self._fname_results_in)  #  这是pareto_iterate的bkingham_iterate.py中代码（lin74有问题）：例如，第2次迭代，此时fname_results_in = 'culled_001.out'；第3次迭代，此时fname_results_in = 'culled_002.out'，以此类推
                        #  正确的应该是Imps_MgO_serial_iterate\buckingham_iterate.py中的代码line90行。此时应该理解为从上一次的culled_00x.out中拟合kde：例如，第2次迭代，此时fname_results_in = 'culled_000.out'；第3次迭代，此时fname_results_in = 'culled_001.out'，以此类推
        # initialize param dict
        param_dict = {}
        for pn in self._param_names:
            param_dict[pn] = None

        # construct free parameter list
        free_param_list = self.get_free_parameter_list()


        # ----------------2021-1-3（测试）: 增加了static变量，增加silverman带宽估计, 强边界判断------------------------------------
        is_good = False
        while not is_good:
            # sample free parameters from kde    # gaussian_kde的resample()重采样，参考链接：https://stackoverflow.com/questions/63178836/scipy-stats-gaussian-kde-to-resample-from-conditional-distribution
            #---------陈洪剑2020-12-9修改：6个自由参数的区间边界条件的判断 —— kde重采样的参数取值应该处于上一次的culled_00x.out对应参数最小值和最大值之间
            condition_nums = 0    # 6个初始值累加和为0
            while condition_nums < len(self._X):   # 例如长度为6：代表6个势函数参数
                flag = [0 for i in range(len(self._X))] # 例如：6个参数的区间都要满足的条件
                self.tmp_free_params = self._kde_kernel.resample(size=1)   #  这里采用gaussian_kde的概率密度函数pdf进行重采样
                # 强化的边界条件判断：基于culled_xxx.out
                for i in range(len(flag)):    # self.tmp_free_params[i][0]等同于self.tmp_free_params[i,0]
                    if self.tmp_free_params[i][0] >= min(self._X[i]) and self.tmp_free_params[i][0] <= max(self._X[i]):
                        flag[i] = 1

                # # 2021-1-3：弱化的边界条件判断
                # for i, free_param in enumerate(free_param_list):    # self.tmp_free_params[i][0]等同于self.tmp_free_params[i,0]
                #     if (self.tmp_free_params[i][0] >= self._param_info[free_param]['info']['a']) and (self.tmp_free_params[i][0] <= self._param_info[free_param]['info']['b']):
                #         flag[i] = 1
                condition_nums = sum(flag)
            # ---------陈洪剑2020-12-9修改

            # （1）free parameter 自由参数的赋值
            for i,pn in enumerate(free_param_list):
                param_dict[pn] = self.tmp_free_params[i,0]  #  重采样后，对参数中的自由参数对应的默认值(一开始都是None)进行重新赋值

            # （2）static variables -------- "static"静态固定参数的赋值
            # for pn in self._param_names:
            #     if self._param_info[pn]['type'] == 'static':
            #         param_dict[pn] = self._param_info[pn]['info']['value']  # 2021-1-3

        # ----------------2021-1-3（测试）: 增加了static变量，增加silverman带宽估计, 强边界判断------------------------------------




            # check parameter constraints
            if param_dict['Guage_S'] == 0:
                is_good = False
            else:
                is_good =True

        return param_dict  # 更新势能参数的具体取值

    def _log(self,msg):
        print(msg)
        self._f_log.write(msg+'\n')    #  self._f_log.name = 'pyposmat.log'



# +++++++++++++++++++++++++++++++++++++ （3）势函数参数采样 + lammps计算代码 +++++++++++++++++++++++++++++++++++

    # 对参数空间进行采样
    def sample_parameter_space(self,
                               n_simulations,
                               fname_results='results.out',  #  模拟结果放置位置的文件名：fname_results默认参数值为'results.out'
                               sampler_type=None,
                               fname_results_in=None):
        """
        Parameters:
        n_simulations - number of simulations
        fname_results - filename of where to put simulation results - 模拟结果放置位置的文件名
        sampler_type - supported_types: uniform, kde
        fname_results_in - required for kde  - kde所需
        """

        start_sim_id = 0          # initialized to normally start at 0
        write_header_files = True # initialized to normally writer headers
        self._results = None      # initialize results
        self._kde_kernel = None   # initialize

        f = open(fname_results, 'w')  # 打开'results_000.out'文件，准备写入数据

        if fname_results_in is not None:   #  例如，第2次迭代，此时fname_results_in = 'culled_001.out'；第3次迭代，此时fname_results_in = 'culled_002.out'，以此类推
            self._fname_results_in = fname_results_in  # 例如第2次迭代，self._fname_results_in = 'culled_001.out'

        if sampler_type is not None:  # 为第2次迭代而设置的(即'kde')
            self._sampler_type = sampler_type  # 第2次迭代为kde

        # if self._is_restart is False:   #  self._is_restart一开始默认值为False
        #     self._log("No restart, starting from sim_id:0")
        #     f = open(fname_results,'w')    #  打开'results_000.out'文件，准备写入数据
        else:
            # restart requested, attempt to restart simulation
            self._log("------------------ Attempting simulation restart, but this method wasn't implemented!------------------")


        # header line strings
        header_line = ", ".join(['sim_id'] + self._names)    #  比如results_000.out文件的标头（第一行）：'sim_id, chrg_Mg, chrg_O, MgMg_A, MgMg_rho, MgMg_C, MgO_A, MgO_rho, MgO_C, OO_A, OO_rho, OO_C, MgO_NaCl.a0, MgO_NaCl.c11, MgO_NaCl.c12, MgO_NaCl.c44, MgO_NaCl.B, MgO_NaCl.G, MgO_NaCl.fr_a, MgO_NaCl.fr_c, MgO_NaCl.sch, MgO_NaCl.001s, MgO_NaCl.a0.err, MgO_NaCl.c11.err, MgO_NaCl.c12.err, MgO_NaCl.c44.err, MgO_NaCl.B.err, MgO_NaCl.G.err, MgO_NaCl.fr_a.err, MgO_NaCl.fr_c.err, MgO_NaCl.sch.err, MgO_NaCl.001s.err'
        types_line = ", ".join(['sim_id'] + self._types)     #  比如results_000.out文件的第二行 ： 'sim_id, param, param, param, param, param, param, param, param, param, param, param, qoi, qoi, qoi, qoi, qoi, qoi, qoi, qoi, qoi, qoi, err, err, err, err, err, err, err, err, err, err'

        # write headers
        if write_header_files:    #  把header_line + types_line内容写入文件中前2行。f.name = 'results_000.out'
            f.write(header_line + "\n")
            f.write(types_line + "\n")

        # f.close()  # 写入前面2行的内容


        # do simulations       #  只有'uniform'和'kde'采样2种方式。第一次迭代时，是用uniform采样，从第2次开始全部使用kde采样
        # df = pd.read_csv(fname_results)  # 创建dataframe格式

        # if os.path.exists(os.path.join(os.getcwd(), 'out_data')):
        #     pass
        # else:
        #     os.mkdir(os.path.join(os.getcwd(), 'out_data'))

        for sim_id in range(start_sim_id, n_simulations):   # start_sim_id = 0， n_simulations = 100


            # self.qois_pre_dict = OrderedDict()

            # 3个task文件任务状态管理【TODO: 考虑放在配置文件中？，根据实际情况，用户自定义设置】
            # task_state_dict = {     'elastic_task':      False,
            #                         'Ec_script_task':    False,
            #                         'Ec_van_f_task':     False
            #                         }    # 任务状态码：初始状态默认均为False   2021-3-3

            task_state_dict = {'code': 'No',
                               'elastic_task':      False,
                               'Ec_script_task':    False,
                               'Ec_van_f_task':     False
                               }

            # -------------------------------------------------------------------- #
            # lammps的3个任务task文件名: ['elastic', 'Ec_script', 'Ec_van_f']         #
            # -------------------------------------------------------------------- #

            # while ((task_state_dict['elastic_task'] == False) or (task_state_dict['Ec_script_task'] == False) or (task_state_dict['Ec_van_f_task'] == False)):
            while task_state_dict['code'] != 'Yes':
                self.qois_pre_dict = OrderedDict()
                
                self.param_dict = None
                # -----------每个sim_id需要-----------------
                self.data = None
                self.line_values = None
                # -----------每个sim_id需要-----------------

                self.results_param = None
                self.results_qoi_pre = None
                self.results_qoi_err = None
                self.line_results = None

                # lammps模拟器：初始化时会确认lmp_eam_fs_gen路径检查，并载入弹性常数、空位形成能、Ec结合能等qois任务
                self._lammps_sim = lammps.SimulationManager()

                #------------------弹性常数qois约束---------------------------------------
                # (1)  TODO: task1：弹性常数目标量的计算
                """
                a.计算Elastic任务, 提取出Cij和B的值
                b.检查预测值和参考值的误差(绝对误差）
                c.内部更新param_dict_state_code状态码信息, sim_id也要保持与原来的不变
                """
                Cij_name_list = ['C11', 'C12', 'C13', 'C33', 'C44']
                # 绝对误差百分比：Absolute percentage errors, APE
                Cij_APE = np.array([1 for i in range(len(Cij_name_list))])  # 初始Cxx绝对误差百分比默认是100% (C11, C12, C13, C33, C44)
                B_APE = 1  # 初始B的误差默认是100%
                condit_Cij = (Cij_APE > 0.98).any()  # 如果设置软条件：condit_Cij = sum(Cij_APE) > 0.2 * len(Cij_APE) # array([0.14208595, 0.11515511, 0.28311436, 0.20777986, 0.17128815])
                condit_B = (B_APE > 0.98)
                while (condit_Cij == True) or (condit_B == True):  # 当误差不在28%以内时，重新对势函数的28个参数param_dict进行采样
                    # -------------------用于生成一组势函数的参数----------------------------
                    print('sim_id={}的param_dict生成~~'.format(sim_id))
                    if self._sampler_type == 'uniform':  # 根据sim_id顺序，一次获得一个一组param参数的具体取值（11个参数）
                        self.param_dict = self.get_uniform_sample()  # 例如第一次是经过均匀分布采样后，更新得到一组势能参数的取值。param_dict = {'chrg_Mg': 2.007954471758032, 'chrg_O': -2.007954471758032, 'MgMg_A': 0.0, 'MgMg_rho': 0.5, 'MgMg_C': 0.0, 'MgO_A': 1267.0168364154858, 'MgO_rho': 0.3231985200115992, 'MgO_C': 0.0, 'OO_A': 9880.708358598808, 'OO_rho': 0.15639322860657612, 'OO_C': 54.037963276262374}
                    elif self._sampler_type == 'kde':
                        self.param_dict = self.get_kde_sample()  # 第2次迭代，是通过kde(用高斯核）对前一次迭代中生成的culled_000.out数据进行概率密度估计，对自由参数进行重采样赋值，其他参数用静态值来更新得到一组势能参数的取值 - update: 2020-10-09
                    else:
                        raise PyPosmatError('unknown sampler type, \'{}\''.format(self._sampler_type))

                    # ★每给定一组势函数参数，自动更新势函数列表文件  # 2020-03-03
                    print('sim_id={}的势函数列表文件生成~~'.format(sim_id))  # 每传入一组势函数参数，就更新势函数列表文件
                    self.potential_file_gen(self.param_dict)  # 每传入一组势函数参数，就更新势函数列表文件

                    # TODO: 实现计算elastic任务的Cij_APE和B_APE的方法
                    self._lammps_sim.create_dir("elastic")          # 1.创建target qoi的目: 例如"xxx/structure_db/elastic
                    elastic_task = lammps.elastic_simulation()      # 2.创建elastic计算的对象
                    elastic_task.lmp_run(self._lammps_sim.dst_dir)  # 3.运行该目录下的.sh脚本的lammps计算   #  确保这里lammps_sim.dst_dir = "xxx/structure_db/elastic"
                    c11_hat, c12_hat, c44_hat, c33_hat, c13_hat, b_hat = elastic_task.get_value() # 获取一组势能参数后，得到的6个qoi目标量
                    # 返回：self.C11, self.C12, self.C44, self.C33, self.C13, self.B
                    # task1中目标量预测值和参考值的误差(绝对误差百分比APE）
                    c11_ape = abs((c11_hat - DFT_qois_dict["C11"]) / DFT_qois_dict["C11"])  # c11的绝对值误差百分比
                    c12_ape = abs((c12_hat - DFT_qois_dict["C12"]) / DFT_qois_dict["C12"])
                    c44_ape = abs((c44_hat - DFT_qois_dict["C44"]) / DFT_qois_dict["C44"])
                    c33_ape = abs((c33_hat - DFT_qois_dict["C33"]) / DFT_qois_dict["C33"])
                    c13_ape = abs((c13_hat - DFT_qois_dict["C13"]) / DFT_qois_dict["C13"])


                    Cij_APE = np.array([c11_ape, c12_ape, c44_ape, c33_ape, c13_ape])
                    # Cij_APE = np.array([abs((c11 - DFT_qois_dict["C11"]) / DFT_qois_dict["C11"]),
                    #            abs((c12 - DFT_qois_dict["C12"]) / DFT_qois_dict["C12"]),
                    #            abs((c13 - DFT_qois_dict["C13"]) / DFT_qois_dict["C13"]),
                    #            abs((c33 - DFT_qois_dict["C33"]) / DFT_qois_dict["C33"]),
                    #            abs((c44 - DFT_qois_dict["C44"]) / DFT_qois_dict["C44"])])
                    B_exp_ref = 36.9   # 参考B的Exp的实验值：36.9
                    B_APE = abs((b_hat - B_exp_ref) / B_exp_ref)
                    condit_Cij = (Cij_APE > 0.98).any()  # 如果设置软条件：condit_Cij = sum(Cij_APE) > 0.2 * len(Cij_APE) # array([0.14208595, 0.11515511, 0.28311436, 0.20777986, 0.17128815])
                    condit_B = (B_APE > 0.98)

                    # 提取task1中的目标量预测值
                    self.qois_pre_dict['C11'] = c11_hat
                    self.qois_pre_dict['C12'] = c12_hat
                    self.qois_pre_dict['C13'] = c13_hat
                    self.qois_pre_dict['C33'] = c33_hat
                    self.qois_pre_dict['C44'] = c44_hat
                    # self.qois_pre_dict['B']   = b_hat     # 不拟合体积模量B

                print('++++++++++++++++++(a)++++++++++++++++++++++++')
                print('ok, Elastic筛选的param_dict参数符合task1要求！')
                task_state_dict['elastic_task'] = True
                print('C11={}, C12={}, C44={}, C33={}, C13={}, B={}'.format(c11_hat, c12_hat, c44_hat, c33_hat, c13_hat, b_hat))
                print('++++++++++++++++++(a)++++++++++++++++++++++++')


                # 执行task2
                while (task_state_dict['elastic_task'] == True) and (task_state_dict['Ec_script_task'] == False):  # 只有当第一个elastic任务条件满足后，才执行第二个任务Ec
                    # -----------------------结构能量差qois的约束----------------------------------
                    # (2) TODO: task2: 结构能量差目标量的计算
                    # 只有task1满足后，才进行task2
                    print()
                    print('将符合Elastic筛选出的self.param_dict参数用于结构能量差的计算')
                    self._lammps_sim.create_dir("Ec_script")                 # 1.创建target qoi目录: 例如"xxx/structure_db/Ec_script
                    import threading
                    Ec_task = lammps.Ec_simulation()                         # 2.创建Ec计算的对象
                    print('-------------------这是while中Ec主线程名字是：{}'.format(threading.current_thread().name))
                    import time
                    first = time.time()
                    Ec_task.lmp_run(self._lammps_sim.dst_dir)                # 3.运行该目录下的.sh脚本的lammps计算   # 确保这里lammps_sim.dst_dir = "xxx/structure_db/Ec_script"
                    print('主while中Ec的主线程{}结束了，用时{}s'.format(threading.current_thread().name, time.time() - first))

                    a_hcp, Ec_value, a_bcc, a_fcc, hcp_bcc_value, hcp_fcc_value =  Ec_task.get_value() # 4.得到一个晶格常数和结合能的值



                    # task2中目标量预测值和参考值的误差(绝对误差百分比APE）
                    latt_a_APE  = np.array([abs((a_hcp - DFT_qois_dict["a"]) / DFT_qois_dict["a"])])
                    Ec_APE      = np.array([abs((Ec_value - DFT_qois_dict["Ec"]) / DFT_qois_dict["Ec"])])
                    hcp_bcc_APE = np.array([abs((hcp_bcc_value - DFT_qois_dict["deta_E_HB"]) / DFT_qois_dict["deta_E_HB"])])
                    hcp_fcc_APE = np.array([abs((hcp_fcc_value - DFT_qois_dict["deta_E_HF"]) / DFT_qois_dict["deta_E_HF"])])

                    condit_HB = (hcp_bcc_value > 0)
                    condit_HF = (hcp_fcc_value > 0)

                    #----------------a_hcp，a_bcc和a_fcc的晶格边界剔除（结构能量差的晶格）上下边界------------
                    cond_a_hcp = ((a_hcp < 3.48) and (a_hcp > 2.9))       #  2.90 < a_hcp < 3.68
                    cond_a_bcc = ((a_bcc < 3.78) and (a_bcc > 3.10))       #  3.10 < a_bcc < 3.88
                    cond_a_fcc = ((a_fcc < 5.18) and (a_fcc > 4.20))       #  4.10 < a_fcc < 4.88

                    # condit_a = (latt_a_APE < 0.68)
                    #---------------------------------
                    # condit_Ec = (Ec_APE < 0.4)    # 暂时省略条件 and (condit_Ec == True)
                    #---------------------------------
                    # condit_hcp_bcc = (hcp_bcc_APE < 0.75)
                    # condit_hcp_fcc = (hcp_fcc_APE < 0.98)
                    # if (condit_HB == True) and (condit_HF == True) and (condit_hcp_bcc == True) and (condit_hcp_fcc == True) and (condit_a == True) and (condit_Ec == True):
                    if ((condit_HB == True) and (condit_HF == True) and (cond_a_hcp == True) and (cond_a_bcc == True) and (cond_a_fcc == True)):
                        # 提取task2中的目标量预测值
                        self.qois_pre_dict['a'] = a_hcp
                        self.qois_pre_dict['Ec'] = Ec_value
                        self.qois_pre_dict['deta_E_HB'] = hcp_bcc_value
                        self.qois_pre_dict['deta_E_HF'] = hcp_fcc_value
                        print('----(b)--------' * 4)
                        print('ok, task1筛选的参数符合task2要求！')
                        task_state_dict['Ec_script_task'] = True
                        print('----(b)--------' * 4)
                    else:  # 如果只符合task1，不符合task2，则退出当前的while循环
                        task_state_dict['elastic_task'] = False
                        task_state_dict['Ec_script_task'] = False
                        task_state_dict['code'] = 'No'
                        break



                while (task_state_dict['elastic_task'] == True) and (task_state_dict['Ec_script_task'] == True) and (task_state_dict['Ec_van_f_task'] == False):
                    # -----------------------空位形成能qois的约束----------------------------------
                    # (3) TODO: task3：空位形成能目标量的计算
                    # 只有task1和task2满足后，才进行task3

                    print()
                    print('将符合Elastic和Ec筛选出的参数用于deta_E_vf计算')
                    self._lammps_sim.create_dir("E_van_f")  # 1.创建target qoi目录: 例如"xxx/structure_db/E_van_f
                    Evf_task = lammps.Evf_simulation()  # 2.创建Evf计算的对象
                    Evf_task.lmp_run(
                        self._lammps_sim.dst_dir)  # 3.运行该目录下的.sh脚本的lammps计算   # 确保这里lammps_sim.dst_dir = "xxx/structure_db/E_van_f"
                    Evf_value = Evf_task.get_value()  # 4.得到空位形成能Evf的值

                    # task3中目标量预测值和参考值的误差(绝对误差百分比APE）
                    Evf_APE = np.array(
                        abs((Evf_value - DFT_qois_dict['deta_E_vf']) / DFT_qois_dict['deta_E_vf']))

                    # condit_Evf = Evf_APE < 0.6

                    if (Evf_value > 0):  # 只要空位形成能大于0就行
                        print('----(c)--------' * 4)
                        print('ok, task1和task2筛选的参数符合task3要求！')
                        # 提取task3中的目标量预测值
                        self.qois_pre_dict['deta_E_vf'] = Evf_value
                        task_state_dict['Ec_van_f_task'] = True
                        task_state_dict['code'] = 'Yes'
                        print('----(c)--------' * 4)
                    else:  # 如果只符合task1和task2，不符合task3，则退出当前的while循环
                        task_state_dict['elastic_task'] = False
                        task_state_dict['Ec_script_task'] = False
                        task_state_dict['Ec_van_f_task'] = False
                        task_state_dict['code'] = 'No'
                        break


                del self._lammps_sim
                gc.collect()

            #------------------判断task1,task2和task3采样后的状态是否是正确的！--------------------------------
            if  (task_state_dict['code'] == 'Yes') and (task_state_dict['elastic_task'] == True) and (
                        task_state_dict['Ec_script_task'] == True) and (
                        task_state_dict['Ec_van_f_task'] == True):


                print('》》》》》》》》》》》》》》》》out数据采集的sim_id = {}'.format(sim_id))

                # --------------------符合task1，task2和task3的势函数参数，将其用于计算目标量qois和误差error-----------------------------------
                #  param_qois_errs_collect函数返回值：self._names, self._types, results。其中results由list(results_param) + list(results_qoi) + list(results_err)组成
                n, t, v = self.param_qois_errs_collect(sim_id, self.param_dict)  # modified by hjhcen 2021-03-16:
                # self.line_values = v
                # self.data = [sim_id] + self.line_values   #  sim_id + 数据值  【注：对应的是.out文件中的一行数据】
                self.data = [sim_id] + v   #  sim_id + 数据值  【注：对应的是.out文件中的一行数据】
                self._log("样本{}计算完并写入xxx.out文件".format(sim_id))   #  在屏幕上打印采样的sim_id索引。等同于打印print("{}".format(sim_id))，第一次sim_id为0，第2次为1，。。。
                print('筛选的sim_id={}的势参数param_dict = \n{}'.format(sim_id, self.param_dict))
                data_line = ", ".join(str(s) for s in self.data)  # 2021-3-16更新：写入.out文件的有47列内容，以“,”分隔开。
                print('+++++++++++++++++++os.getcwd():{}'.format(os.getcwd()))  # /home/hjchen/simulation_projects/Mg/new_qois_and_APE_version/test
                print('data_line = ', data_line)
                print('data_line[27:37] = ', ",".join(str(s) for s in self.data).split(',')[27:37])

                # df.loc[df.index.size] = self.data   # 往dataframe的最后增加一行数据

                print('--------------------step(1): sim_id = {}写入out后----------------------------------------------'.format(sim_id))
                f.write(data_line + '\n')

                del task_state_dict
                gc.collect()

            else:
                print('+++++++++++++++++++++++++++++++++sim_id={}采样错误!'.format(sim_id))
                print('task_state_dict = ', task_state_dict)
                break



        f.close()







# +++++++++++++++++++++++++++ lammps的计算过程 +++++++++++++++++++++++++++++++++++++++++++++++++++++
    def param_qois_errs_collect(self, sim_id, param_dict):
        self.qoi_calc_values, self.qois_calc_errs = self.calculate_qois(self.qois_pre_dict)

        self.results_param = [param_dict[p] for p in list(config.parameter_distribution.keys())]
        print('self.qoi_calc_values = ', self.qoi_calc_values)
        print('self.qois_calc_errs = ', self.qois_calc_errs)
        self.results_qoi_pre = [self.qoi_calc_values[q] for q in self._qoi_names]  # self._qoi_names = ['MgO_NaCl.a0','MgO_NaCl.c11','MgO_NaCl.c12','MgO_NaCl.c44','MgO_NaCl.B','MgO_NaCl.G','MgO_NaCl.fr_a','MgO_NaCl.fr_c','MgO_NaCl.sch','MgO_NaCl.001s']
        self.results_qoi_err = [self.qois_calc_errs[q] for q in self._qoi_names]

        self.line_results = list(self.results_param) + list(self.results_qoi_pre) + list(self.results_qoi_err)    #  结果的数据组成2020-12-18: 28个参数 + 7个qoi + 7个err = 42个
        print('--------------------step(1): sim_id = {}写入out前----------------------------------------------'.format(sim_id))
        print('sim_id = ', sim_id)
        print('param_dict = ', param_dict)
        print('self.results_param = ', self.results_param)
        print('self.results_qoi_pre = ', self.results_qoi_pre)
        print('self.results_qoi_err = ', self.results_qoi_err)

        return self._names, self._types, self.line_results




    def calculate_qois(self, var_dict):

        self._qoi_values = copy.deepcopy(var_dict)  # 即3者是相等的：self.qois_pre_dict = var_dict = self._qoi_values
        self._qoi_errors = {}
        # 需要注意字典是无序的，因此字典类型预测值self.qoi_pre_dict和字典类型参考值self.qois_ref_dict可能存在key和value不是一一对应的
        # 存的是第一性原理计算的参考值
        self.qois_ref_dict = OrderedDict()
        for key, v in zip(qoi_name_list, qoi_value_list):
            self.qois_ref_dict[key] = v
        try:
            for k in self._qoi_names:
                self._qoi_errors[k] = var_dict[k] - self.qois_ref_dict[k]

        except Exception as e:
            print('计算qois.err出错：', e)
            self._qoi_errors = None

        return self._qoi_values, self._qoi_errors





# if __name__ == '__main__':
#     root = os.path.abspath(os.path.join(os.getcwd(), ".."))
#     print('当前文件路径：', root)
#     f = open(os.path.join(root, 'lmp_eam_fs_gen', 'Zn_20201101chj.eam.fs'), 'r')
#     print(f.name)