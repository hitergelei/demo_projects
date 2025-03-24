import numpy as np
import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

parameter_distribution = OrderedDict()


# 自定义势函数参数
param_dict ={'ZBL_B0': 7.166325242535745, 'ZBL_B1': -3.3938997010255667, 'ZBL_B2': -0.10504397794175134, 'ZBL_B3': 0.07254200501053336, 'Guage_S': 0.6027907819496395, 'Guage_C': 0.3993839150567203, 'den_A1': 0.025581841572739544, 'den_A2': -0.0013604163682887117, 'den_A3': 0.005113005645510601, 'den_A4': -0.10081312330587366, 'den_A5': 0.05052399988866524, 'den_A6': -0.021653257558713723, 'den_A7': 0.011855219254877737, 'den_A8': 0.16253529444896747, 'den_A9': -0.3655690247411456, 'den_A10': 0.1756477511959416, 'P_a1': -0.008088946099782505, 'P_a2': 0.022666538904064823, 'P_a3': 0.02334877201791307, 'P_a4': -0.04062428817888836, 'P_a5': -0.06798681446820046, 'P_a6': 0.0020717454699583793, 'P_a7': 0.2554644626782531, 'P_a8': 0.2654156376791926, 'P_a9': 0.4089577533409682, 'P_a10': -0.14190084410025883}


#---------------定义好一些内存变量--------------------
emb =  [None for i in range(10000)]     # 嵌入能的值
Φ_eff =   [None for i in range(10000)]     # 电子密度函数值
vsum = [None for i in range(10000)]     # 对势函数值
v =    [None for i in range(10000)]     # 乘以r后的对势值

Ak = [None for i in range(10)]
Rk = [None for i in range(10)]
ak = [None for i in range(10)]
rk = [None for i in range(10)]

bzb = [None for i in range(4)]


# -------- ZBL connect parameters                      ZBL连接参数：有4个参数
qele1 = 12.0    # for Zn
qele2 = 12.0    # for Zn
b0 = param_dict['ZBL_B0']     # B0参数
b1 = param_dict['ZBL_B1']    # B1参数
b2 = param_dict['ZBL_B2']   # B2参数
b3 = param_dict['ZBL_B3']     # B3参数

#---------Guage parameters                             规范不变性参数：有2个参数
S = param_dict['Guage_S']     # S参数
C = param_dict['Guage_C']      # C参数

#---------potential parameters                          势函数参数：共22个参数

# (1).电子密度函数的参数（10个需要调参）
# Ak：节点系数
Ak[0]= param_dict['den_A1']
Ak[1]= param_dict['den_A2']
Ak[2]= param_dict['den_A3']
Ak[3]= param_dict['den_A4']
Ak[4]= param_dict['den_A5']
Ak[5]= param_dict['den_A6']
Ak[6]= param_dict['den_A7']
Ak[7]= param_dict['den_A8']
Ak[8]= param_dict['den_A9']
Ak[9]= param_dict['den_A10']

# (2).对势函数的参数（10个需要调参）
# ak：节点系数
ak[0]= param_dict['P_a1']
ak[1]= param_dict['P_a2']
ak[2]= param_dict['P_a3']
ak[3]= param_dict['P_a4']
ak[4]= param_dict['P_a5']
ak[5]= param_dict['P_a6']
ak[6]= param_dict['P_a7']
ak[7]= param_dict['P_a8']
ak[8]= param_dict['P_a9']
ak[9]= param_dict['P_a10']

# (3).嵌入能的参数A和B （2个需要调参）
# A = param_dict['eam_A']    # 嵌入能参数A
# B = param_dict['eam_B']     # 嵌入能参数B

#-------------------------------------------



# Rk：节点位置的固定值（输入值，不需要调参）
Rk[0] = 6.3
Rk[1] = 5.9
Rk[2] = 5.5
Rk[3] = 5.1
Rk[4] = 4.7
Rk[5] = 4.3
Rk[6] = 3.9
Rk[7] = 3.5
Rk[8] = 3.1
Rk[9] = 2.7
# rk：节点位置的固定值（输入值，不需要调参）
rk[0] = 6.3
rk[1] = 5.9
rk[2] = 5.5
rk[3] = 5.1
rk[4] = 4.7
rk[5] = 4.3
rk[6] = 3.9
rk[7] = 3.5
rk[8] = 3.1
rk[9] = 2.7

#--------------------
dρ=2.0000e-0003     # 即drho
dr=6.3000e-0004
#---------------------

def HH(x):   # Hs(x)是单位阶跃函数
    if x > 0: return 1
    else: return 0

# file = open('Mg_chj2021_select_idx(%s)_10_1000_10qois.eam.fs'%idx, 'w')   #  势函数文件列表
file = open('Mg_2021chj.eam.fs', 'w')   #  势函数文件列表
file.write("Mg potential\n")
file.write("Finnis-Sinclair formalism\n")
file.write("writen by hjchen in %s\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
file.write("1  Mg\n")
file.write("10000  2.0000E-0003  10000  6.3000E-0004  6.3000E+0000\n")
file.write("12   2.43050E+0001   3.2090E+0000  hcp\n")  # 把晶格常数写死了就是3.209

#----------------嵌入能的计算
for i in range(0, 10000):
    ρ = float(i) * dρ     # 对应ρ = i * dρ
    if i == 0:
        emb[i] = 0.0e0
    else:
        # 进行规范不变性变换后，嵌入能公式F_eff(ρ) = F(ρ/S) + (C/S)*ρ
        # 其中，F(ρi) = -√ρi 
        emb[i] = - np.sqrt(ρ/S)  + (C/S) * ρ

for i in range(0, int(10000/5)):   # 分成5列写入到势函数文件中
    file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
               format(emb[5 * i], emb[5 * i + 1], emb[5 * i + 2], emb[5 * i + 3],
                      emb[5 * i + 4]))


#----------------电子密度函数Φ(r)的计算
for i in range(0, 10000):
    r = float(i) * dr   # 对应r = i * dr
    # fr1对应的是未进行规范不变性变换的对势函数Φ(r)
    Φ_r = Ak[0] * (Rk[0] - r) ** 3 * HH(Rk[0] - r) + \
          Ak[1] * (Rk[1] - r) ** 3 * HH(Rk[1] - r) + \
          Ak[2] * (Rk[2] - r) ** 3 * HH(Rk[2] - r) + \
          Ak[3] * (Rk[3] - r) ** 3 * HH(Rk[3] - r) + \
          Ak[4] * (Rk[4] - r) ** 3 * HH(Rk[4] - r) + \
          Ak[5] * (Rk[5] - r) ** 3 * HH(Rk[5] - r) + \
          Ak[6] * (Rk[6] - r) ** 3 * HH(Rk[6] - r) + \
          Ak[7] * (Rk[7] - r) ** 3 * HH(Rk[7] - r) + \
          Ak[8] * (Rk[8] - r) ** 3 * HH(Rk[8] - r) + \
          Ak[9] * (Rk[9] - r) ** 3 * HH(Rk[9] - r)
    # fr对应的是规范不变性变换后的对势函数Φ_eff(r)
    Φ_eff[i] = S * Φ_r    # 对应公式：Φ_eff(r) = S * Φ(r)

for i in range(0, int(10000/5)):   # 分成5列写入到势函数文件中
    file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
               format(Φ_eff[5 * i], Φ_eff[5 * i + 1], Φ_eff[5 * i + 2], Φ_eff[5 * i + 3],
                      Φ_eff[5 * i + 4]))

#---------------
r_value = []
pair_potential = []
for i in range(0, 10000):
    r = float(i) * dr
    if r == 0:
        r = 0.1e-12
    if i == 0:
        Zi   = qele1                                    # 对应Zi
        Zj   = qele2                                    # 对应Zj
        ev     = 1.602176565e-19                        # 对应e
        pi     = 3.14159265358979324e0                  # 对应π
        epsil0 = 8.854187817e-12                        # 对应ε0
        bohrad = 0.52917721067e0                       # 波尔半径：0.529埃
        exn    = 0.23e0                                # 幂函数的指数（常数）
        beta   = 1/(4.0*pi*epsil0) * (Zi*Zj*np.power(ev,2)) * 1.0e10/ev
        a      = 0.8854*bohrad/(np.power(Zi, exn) + np.power(Zj, exn))   #  对应公式a = 0.4685335/(Zi^0.23 + Zj^0.23)
        # bzb[0] = -3.19980 / a     #  对应x = rij/a
        # bzb[1] = -0.94229 / a     #  对应x = rij/a
        # bzb[2] = -0.40290 / a     #  对应x = rij/a
        # bzb[3] = -0.20162 / a     #  对应x = rij/a
        v[i]   = beta * (0.18175 * np.exp(-3.19980 * r/a) +
                         0.50986 * np.exp(-0.94229 * r/a) +
                         0.28022 * np.exp(-0.40290 * r/a) +
                         0.02817 * np.exp(-0.20162 * r/a))

    # (1) r < rm时，计算的对势公式是V_ZBL
    if r < 1.0:
        Zi = qele1
        Zj = qele2
        ev = 1.602176565e-19
        pi = 3.14159265358979324e0
        epsil0 = 8.854187817e-12
        bohrad = 0.52917721067e0
        exn = 0.23e0
        # beta的公式
        beta = 1 / (4.0 * pi * epsil0) * (Zi * Zj * np.power(ev, 2)) * 1.0e10 / ev
        a = 0.8854*bohrad/(np.power(Zi, exn) + np.power(Zj, exn))
        # bzb[0] = -3.19980/a
        # bzb[1] = -0.94229/a
        # bzb[2] = -0.40290/a
        # bzb[3] = -0.20162/a
        rinv = 1.0/r
        vsum[i] = beta * rinv * (0.18175 * np.exp(-3.19980 * r/a) +
                                 0.50986 * np.exp(-0.94229 * r/a) +
                                 0.28022 * np.exp(-0.40290 * r/a) +
                                 0.02817 * np.exp(-0.20162 * r/a))

    # (2) rm <= r <= rn时，计算的对势公式是：V_Connect
    elif r >= 1.0 and r < 2.3:
        vsum[i] = np.exp(b0+b1*r+b2*np.power(r,2)+b3*np.power(r,3))

    # (3) r >=rn时，计算的对势公式是：V_Original
    elif r >= 2.3:
        vsum[i] = ak[0] * (rk[0] - r) ** 3 * HH(rk[0] - r) + \
                  ak[1] * (rk[1] - r) ** 3 * HH(rk[1] - r) + \
                  ak[2] * (rk[2] - r) ** 3 * HH(rk[2] - r) + \
                  ak[3] * (rk[3] - r) ** 3 * HH(rk[3] - r) + \
                  ak[4] * (rk[4] - r) ** 3 * HH(rk[4] - r) + \
                  ak[5] * (rk[5] - r) ** 3 * HH(rk[5] - r) + \
                  ak[6] * (rk[6] - r) ** 3 * HH(rk[6] - r) + \
                  ak[7] * (rk[7] - r) ** 3 * HH(rk[7] - r) + \
                  ak[8] * (rk[8] - r) ** 3 * HH(rk[8] - r) + \
                  ak[9] * (rk[9]-r) ** 3 * HH(rk[9] - r) - \
              2.0 * C * (Ak[0] * (Rk[0] - r) ** 3 * HH(Rk[0] - r) +
                         Ak[1] * (Rk[1] - r) ** 3 * HH(Rk[1] - r) +
                         Ak[2] * (Rk[2] - r) ** 3 * HH(Rk[2] - r) +
                         Ak[3] * (Rk[3] - r) ** 3 * HH(Rk[3] - r) +
                         Ak[4] * (Rk[4] - r) ** 3 * HH(Rk[4] - r) +
                         Ak[5] * (Rk[5] - r) ** 3 * HH(Rk[5] - r) +
                         Ak[6] * (Rk[6] - r) ** 3 * HH(Rk[6] - r) +
                         Ak[7] * (Rk[7] - r) ** 3 * HH(Rk[7] - r) +
                         Ak[8] * (Rk[8] - r) ** 3 * HH(Rk[8] - r) +
                         Ak[9] * (Rk[9] - r) ** 3 * HH(Rk[9] - r))

    r_value.append(r)    # 把r存起来
    pair_potential.append(vsum[i])  # 把对势的值存起来（没有乘以r的）
    v[i] = r * vsum[i]

#np.savetxt('r_value.txt', r_value)
#np.savetxt('pair_potential_value.txt', pair_potential)   # 保存没有乘以r的对势值

r_test = []
pair_test = []
for r, pair in zip(r_value, pair_potential):
    if np.isnan(np.array(pair)):
        pass
    else:
        r_test.append(r)
        pair_test.append(pair)

plt.plot(r_test, pair_test)
plt.xlim(1.5, 6)
plt.ylim(-0.5, 3)
plt.show()


for i in range(0, int(10000/5)):  # 分成5列写入到势函数文件中
    file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
               format(v[5 * i], v[5 * i + 1], v[5 * i + 2], v[5 * i + 3],
                      v[5 * i + 4]))

file.close()