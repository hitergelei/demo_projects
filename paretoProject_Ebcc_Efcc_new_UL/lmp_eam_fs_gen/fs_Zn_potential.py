import numpy as np
import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime

#------------------------------------------------------------测试：Mg.eam.fs的生成------------------------------------
#---------------定义好一些内存变量--------------------
emb =  [None for i in range(10000)]     # 嵌入能的值
fr =   [None for i in range(10000)]     # 电子密度函数值
vsum = [None for i in range(10000)]     # 对势函数值
v =    [None for i in range(10000)]     # 乘以r后的对势值

T_ao = [None for i in range(10)]   # Ak
T_ro = [None for i in range(10)]   # Rk
T_ap = [None for i in range(10)]   # ak
T_rp = [None for i in range(10)]   # rk
T_af = [None for i in range(2)]    # A 和 B

bzb = [None for i in range(4)]


# -------- ZBL connect parameters                      ZBL连接参数：有4个参数
qele1 = 12.0    # for Mg
qele2 = 12.0    # for Mg
b0 = 7.13869020382836     # B0参数
b1 = -3.42213335283108    # B1参数
b2 = -0.0631964668839501    # B2参数
b3 = 0.0652873462213261     # B3参数

#---------Guage parameters                             规范不变性参数：有2个参数
T_sss = 0.645841533432715     # S参数
T_ccc = 0.401821332632028      # C参数

#---------potential parameters                          势函数参数：共22个参数

# (1).电子密度函数的参数（10个需要调参）
# Ak：节点系数
T_ao[0]=0.00107141640725646
T_ao[1]=-0.00210005300301463
T_ao[2]=0.0578954145943402
T_ao[3]=-0.113764843494093
T_ao[4]=0.0568853339673111
T_ao[5]=0.00145565626552289
T_ao[6]=-0.00306278411316600
T_ao[7]=0.169636267604889
T_ao[8]=-0.335855111019231
T_ao[9]=0.167896433936514


# (2).对势函数的参数（10个需要调参）
# ak：节点系数
T_ap[0]=-0.000379992731900483
T_ap[1]=-0.00970160325290971
T_ap[2]=0.0223615549371360   # a3
T_ap[3]=0.0102799999816850
T_ap[4]=-0.0652603934277463
T_ap[5]=-0.0541894195773795
T_ap[6]=0.304891344131519
T_ap[7]=0.277105917461963
T_ap[8]=0.386349382979606
T_ap[9]=-0.122340639111536


# (3).嵌入能的参数A和B （2个需要调参）
#T_af[0] = -0.971262913156540e-02    # 嵌入能参数A
#T_af[1] = 0.364558125014771e-05     # 嵌入能参数B

#-------------------------------------------



# Rk：节点位置的固定值（输入值，不需要调参）
T_ro[0] = 6.3
T_ro[1] = 5.9
T_ro[2] = 5.5
T_ro[3] = 5.1
T_ro[4] = 4.7
T_ro[5] = 4.3
T_ro[6] = 3.9
T_ro[7] = 3.5
T_ro[8] = 3.1
T_ro[9]= 2.7
# rk：节点位置的固定值（输入值，不需要调参）
T_rp[0] = 6.3
T_rp[1] = 5.9
T_rp[2] = 5.5
T_rp[3] = 5.1
T_rp[4] = 4.7
T_rp[5] = 4.3
T_rp[6] = 3.9
T_rp[7] = 3.5
T_rp[8] = 3.1
T_rp[9]= 2.7

#--------------------
dp=2.0000e-0003     # 即drho
dr=6.3000E-0004
#---------------------

def HH(x):   # Hs(x)是单位阶跃函数
    if x > 0: return 1
    else: return 0

file = open('Mg_2021test_chj.eam.fs', 'w')   #  势函数文件列表
file.write("Mg potential\n")
file.write("Finnis-Sinclair formalism\n")
file.write("writen by hjchen in %s\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
file.write("1  Mg\n") # Number of Elements, Element symbol
file.write("10000  2.0000E-0003  10000  6.3000E-0004  6.3000E+0000\n") # Nrho, drho, Nr, dr, cutoff
# Nrho: Number of points at which electron density is evaluated  -  评估电子密度的点数
# drho: distance between points where the electron density is evaluated - 计算电子密度的点之间的距离
# Nr: Number of points at which the interatomic potential and embedding function is evaluated - 评估原子间电势和嵌入函数的点数
# dr: distance between points where the interatomic potential and embedding function is evaluated - 评估原子间势和嵌入函数的点之间的距离
# ctoff: cutoff distance for all functions(Angstroms) - 所有函数的截断距离（埃）
file.write("12   2.43050E+0001   3.2090E+0000  hcp\n")  # 原子序数， 相对原子质量， 晶格常数， hcp

#----------------嵌入能的计算
for i in range(0, 10000):
    r = float(i) * dp     # 对应ρ = i * dρ
    if i == 0:
        emb[i] = 0.0e0
    else:
        # 进行规范不变性变换后，嵌入能公式F_eff(ρ) = F(ρ/S) + (C/S)*ρ
        # 其中，F(ρi) = -√ρi + A*ρi^2 + B*ρi^4
        emb[i] = - np.sqrt(r/T_sss) + (T_ccc/T_sss) * r

for i in range(0, int(10000/5)):   # 分成5列写入到势函数文件中
    file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
               format(emb[5 * i], emb[5 * i + 1], emb[5 * i + 2], emb[5 * i + 3],
                      emb[5 * i + 4]))



#----------------电子密度函数Φ(r)的计算
for i in range(0, 10000):
    r = float(i) * dr   # 对应r = i * dr
    # fr1对应的是未进行规范不变性变换的对势函数Φ(r)
    fr1 = T_ao[0] * (T_ro[0] - r) ** 3 * HH(T_ro[0] - r) + \
          T_ao[1] * (T_ro[1] - r) ** 3 * HH(T_ro[1] - r) + \
          T_ao[2] * (T_ro[2] - r) ** 3 * HH(T_ro[2] - r) + \
          T_ao[3] * (T_ro[3] - r) ** 3 * HH(T_ro[3] - r) + \
          T_ao[4] * (T_ro[4] - r) ** 3 * HH(T_ro[4] - r) + \
          T_ao[5] * (T_ro[5] - r) ** 3 * HH(T_ro[5] - r) + \
          T_ao[6] * (T_ro[6] - r) ** 3 * HH(T_ro[6] - r) + \
          T_ao[7] * (T_ro[7] - r) ** 3 * HH(T_ro[7] - r) + \
          T_ao[8] * (T_ro[8] - r) ** 3 * HH(T_ro[8] - r) + \
          T_ao[9] * (T_ro[9] - r) ** 3 * HH(T_ro[9] - r)
    # fr对应的是规范不变性变换后的对势函数Φ_eff(r)
    fr[i] = T_sss * fr1    # 对应公式：Φ_eff(r) = S * Φ(r)

for i in range(0, int(10000/5)):   # 分成5列写入到势函数文件中
    file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
               format(fr[5 * i], fr[5 * i + 1], fr[5 * i + 2], fr[5 * i + 3],
                      fr[5 * i + 4]))

#---------------
r_value = []
pair_potential = []
for i in range(0, 10000):
    r = float(i) * dr
    if r == 0:
        r = 0.1e-30
    if i == 0:
        zed1   = qele1                                  # 对应Zi
        zed2   = qele2                                  # 对应Zj
        ev     = 1.602176565e-19                        # 对应e
        pi     = 3.14159265358979324e0                  # 对应π
        epsil0 = 8.854187817e-12                        # 对应ε0
        bohrad = 0.52917721067e0                       # 波尔半径：0.529埃
        exn    = 0.23e0                                # 幂函数的指数（常数）
        beta   = (zed1*zed2*ev*ev)/(4.0*pi*epsil0)*1.0e10/ev
        rs     = 0.8854*bohrad/(zed1**exn +zed2**exn)   #  对应公式a = 0.4685335/(Zi^0.23 + Zj^0.23)
        bzb[0] = -3.19980 / rs     #  对应x = rij/a
        bzb[1] = -0.94229 / rs     #  对应x = rij/a
        bzb[2] = -0.40290 / rs     #  对应x = rij/a
        bzb[3] = -0.20162 / rs     #  对应x = rij/a
        v[i]   = beta * (0.18175 * np.exp(bzb[0] * r) +
                         0.50986 * np.exp(bzb[1] * r) +
                         0.28022 * np.exp(bzb[2] * r) +
                         0.02817 * np.exp(bzb[3] * r))

    if r < 1.0:
        zed1 = qele1
        zed2 = qele2
        ev = 1.602176565e-19
        pi = 3.14159265358979324e0
        epsil0 = 8.854187817e-12
        bohrad = 0.52917721067e0
        exn = 0.23e0
        beta = (zed1*zed2*ev*ev)/(4.0*pi*epsil0)*1.0e10/ev
        rs = 0.8854*bohrad/(zed1**exn +zed2**exn)
        bzb[0] = -3.19980/rs
        bzb[1] = -0.94229/rs
        bzb[2] = -0.40290/rs
        bzb[3] = -0.20162/rs
        rinv = 1.0/r
        vsum[i] = beta * rinv * (0.18175 * np.exp(bzb[0] * r) +
                                 0.50986 * np.exp(bzb[1] * r) +
                                 0.28022 * np.exp(bzb[2] * r) +
                                 0.02817 * np.exp(bzb[3] * r))

    elif r >= 1.0 and r < 2.3:
        vsum[i] = np.exp(b0+b1*r+b2*r**2.0+b3*r**3.0)

    elif r >= 2.3:
        vsum[i] = T_ap[0] * (T_rp[0] - r) ** 3 * HH(T_rp[0] - r) + \
        T_ap[1] * (T_rp[1] - r) ** 3 * HH(T_rp[1] - r) + \
        T_ap[2] * (T_rp[2] - r) ** 3 * HH(T_rp[2] - r) + \
        T_ap[3] * (T_rp[3] - r) ** 3 * HH(T_rp[3] - r) + \
        T_ap[4] * (T_rp[4] - r) ** 3 * HH(T_rp[4] - r) + \
        T_ap[5] * (T_rp[5] - r) ** 3 * HH(T_rp[5] - r) + \
        T_ap[6] * (T_rp[6] - r) ** 3 * HH(T_rp[6] - r) + \
        T_ap[7] * (T_rp[7] - r) ** 3 * HH(T_rp[7] - r) + \
        T_ap[8] * (T_rp[8] - r) ** 3 * HH(T_rp[8] - r) + \
        T_ap[9] * (T_rp[9]-r) ** 3 * HH(T_rp[9] - r) - \
 2.0 * T_ccc * (T_ao[0] * (T_ro[0] - r) ** 3 * HH(T_ro[0] - r) +
                T_ao[1] * (T_ro[1] - r) ** 3 * HH(T_ro[1] - r) +
                T_ao[2] * (T_ro[2] - r) ** 3 * HH(T_ro[2] - r) +
                T_ao[3] * (T_ro[3] - r) ** 3 * HH(T_ro[3] - r) +
                T_ao[4] * (T_ro[4] - r) ** 3 * HH(T_ro[4] - r) +
                T_ao[5] * (T_ro[5] - r) ** 3 * HH(T_ro[5] - r) +
                T_ao[6] * (T_ro[6] - r) ** 3 * HH(T_ro[6] - r) +
                T_ao[7] * (T_ro[7] - r) ** 3 * HH(T_ro[7] - r) +
                T_ao[8] * (T_ro[8] - r) ** 3 * HH(T_ro[8] - r) +
                T_ao[9] * (T_ro[9] - r) ** 3 * HH(T_ro[9] - r))

    r_value.append(r)    # 把r存起来
    pair_potential.append(vsum[i])  # 把对势的值存起来（没有乘以r的）
    v[i] = r * vsum[i]

np.savetxt('r_value.txt', r_value)
np.savetxt('pair_potential_value.txt', pair_potential)   # 保存没有乘以r的对势值

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


for i in range(0, int(10000 / 5)):  # 分成5列写入到势函数文件中
    file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
               format(v[5 * i], v[5 * i + 1], v[5 * i + 2], v[5 * i + 3],
                      v[5 * i + 4]))

file.close()
#
#
#
#
#
#

#---------------------------------------------测试：Mg_20190916.eam.fs的生成--------------------------
# import numpy as np
# import copy
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from datetime import datetime
#
# #---------------定义好一些内存变量--------------------
# emb =  [None for i in range(10000)]     # 嵌入能的值
# fr =   [None for i in range(10000)]     # 电子密度函数值
# vsum = [None for i in range(10000)]     # 对势函数值
# v =    [None for i in range(10000)]     # 乘以r后的对势值
#
# T_ao = [None for i in range(10)]   # Ak
# T_ro = [None for i in range(10)]   # Rk
# T_ap = [None for i in range(10)]   # ak
# T_rp = [None for i in range(10)]   # rk
# T_af = [None for i in range(2)]    # A 和 B
#
# bzb = [None for i in range(4)]
#
#
# # -------- ZBL connect parameters                      ZBL连接参数：有4个参数
# qele1 = 12.0    # for Mg
# qele2 = 12.0    # for Mg
# b0 = 0.704664243977151e+01     # B0参数
# b1 = -0.333010469851498e+01    # B1参数
# b2 = -0.619555583046675e-01    # B2参数
# b3 = 0.640655473828010e-01    # B3参数
#
# #---------Guage parameters                             规范不变性参数：有2个参数
# T_sss = 0.644973966426141e+00     # S参数
# T_ccc = 0.401551356125883e+00      # C参数
#
# #---------potential parameters                          势函数参数：共22个参数
#
# # (1).电子密度函数的参数（10个需要调参）
# # Ak：节点系数
# T_ao[0]=0.103505961304011e-02
# T_ao[1]=-0.200456259837736e-02
# T_ao[2]=0.582304132777502e-01
# T_ao[3]=-0.114559033299128e+00
# T_ao[4]=0.572667170534239e-01
# T_ao[5]=0.131921246010663e-03
# T_ao[6]=-0.210012600486547e-03
# T_ao[7]=0.175729529142241e+00
# T_ao[8]=-0.370683599716195e+00
# T_ao[9]=0.214070465439199e+01
#
#
# # (2).对势函数的参数（10个需要调参）
# # ak：节点系数
# T_ap[0]=-0.259183792597927e-03
# T_ap[1]=-0.904891038338592e-02
# T_ap[2]=0.218862253193047e-01
# T_ap[3]=0.586212126710029e-02
# T_ap[4]=-0.627354410264580e-01
# T_ap[5]=-0.409394774379268e-01
# T_ap[6]=0.282588362952282e+00
# T_ap[7]=0.297152769812915e+00
# T_ap[8]=0.566647129926957e+00
# T_ap[9]=0.981038501882758e-01
#
#
# # (3).嵌入能的参数A和B （2个需要调参）
# #T_af[0] = -0.971262913156540e-02    # 嵌入能参数A
# #T_af[1] = 0.364558125014771e-05     # 嵌入能参数B
#
# #-------------------------------------------
#
#
#
# # Rk：节点位置的固定值（输入值，不需要调参）
# T_ro[0] = 6.3
# T_ro[1] = 5.9
# T_ro[2] = 5.5
# T_ro[3] = 5.1
# T_ro[4] = 4.7
# T_ro[5] = 4.3
# T_ro[6] = 3.9
# T_ro[7] = 3.5
# T_ro[8] = 3.1
# T_ro[9]= 2.7
# # rk：节点位置的固定值（输入值，不需要调参）
# T_rp[0] = 6.3
# T_rp[1] = 5.9
# T_rp[2] = 5.5
# T_rp[3] = 5.1
# T_rp[4] = 4.7
# T_rp[5] = 4.3
# T_rp[6] = 3.9
# T_rp[7] = 3.5
# T_rp[8] = 3.1
# T_rp[9]= 2.7
#
# #--------------------
# dp=2.0000e-0003     # 即drho
# dr=6.3000E-0004
# #---------------------
#
# def HH(x):   # Hs(x)是单位阶跃函数
#     if x > 0: return 1
#     else: return 0
#
# file = open('Mg_2021test_chj.eam.fs', 'w')   #  势函数文件列表
# file.write("Mg potential\n")
# file.write("Finnis-Sinclair formalism\n")
# file.write("writen by hjchen in %s\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# file.write("1  Mg\n") # Number of Elements, Element symbol
# file.write("10000  2.0000E-0003  10000  6.3000E-0004  6.3000E+0000\n") # Nrho, drho, Nr, dr, cutoff
# # Nrho: Number of points at which electron density is evaluated  -  评估电子密度的点数
# # drho: distance between points where the electron density is evaluated - 计算电子密度的点之间的距离
# # Nr: Number of points at which the interatomic potential and embedding function is evaluated - 评估原子间电势和嵌入函数的点数
# # dr: distance between points where the interatomic potential and embedding function is evaluated - 评估原子间势和嵌入函数的点之间的距离
# # ctoff: cutoff distance for all functions(Angstroms) - 所有函数的截断距离（埃）
# file.write("12   2.43050E+0001   3.2090E+0000  hcp\n")  # 原子序数， 相对原子质量， 晶格常数， hcp
#
# #----------------嵌入能的计算
# for i in range(0, 10000):
#     r = float(i) * dp     # 对应ρ = i * dρ
#     if i == 0:
#         emb[i] = 0.0e0
#     else:
#         # 进行规范不变性变换后，嵌入能公式F_eff(ρ) = F(ρ/S) + (C/S)*ρ
#         # 其中，F(ρi) = -√ρi + A*ρi^2 + B*ρi^4
#         emb[i] = - np.sqrt(r/T_sss) + (T_ccc/T_sss) * r
#
# for i in range(0, int(10000/5)):   # 分成5列写入到势函数文件中
#     file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
#                format(emb[5 * i], emb[5 * i + 1], emb[5 * i + 2], emb[5 * i + 3],
#                       emb[5 * i + 4]))
#
#
#
# #----------------电子密度函数Φ(r)的计算
# for i in range(0, 10000):
#     r = float(i) * dr   # 对应r = i * dr
#     # fr1对应的是未进行规范不变性变换的对势函数Φ(r)
#     fr1 = T_ao[0] * (T_ro[0] - r) ** 3 * HH(T_ro[0] - r) + \
#           T_ao[1] * (T_ro[1] - r) ** 3 * HH(T_ro[1] - r) + \
#           T_ao[2] * (T_ro[2] - r) ** 3 * HH(T_ro[2] - r) + \
#           T_ao[3] * (T_ro[3] - r) ** 3 * HH(T_ro[3] - r) + \
#           T_ao[4] * (T_ro[4] - r) ** 3 * HH(T_ro[4] - r) + \
#           T_ao[5] * (T_ro[5] - r) ** 3 * HH(T_ro[5] - r) + \
#           T_ao[6] * (T_ro[6] - r) ** 3 * HH(T_ro[6] - r) + \
#           T_ao[7] * (T_ro[7] - r) ** 3 * HH(T_ro[7] - r) + \
#           T_ao[8] * (T_ro[8] - r) ** 3 * HH(T_ro[8] - r) + \
#           T_ao[9] * (T_ro[9] - r) ** 3 * HH(T_ro[9] - r)
#     # fr对应的是规范不变性变换后的对势函数Φ_eff(r)
#     fr[i] = T_sss * fr1    # 对应公式：Φ_eff(r) = S * Φ(r)
#
# for i in range(0, int(10000/5)):   # 分成5列写入到势函数文件中
#     file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
#                format(fr[5 * i], fr[5 * i + 1], fr[5 * i + 2], fr[5 * i + 3],
#                       fr[5 * i + 4]))
#
# #---------------
# r_value = []
# pair_potential = []
# for i in range(0, 10000):
#     r = float(i) * dr
#     if r == 0:
#         r = 0.1e-24
#         # pass
#     if i == 0:
#         zed1   = qele1                                  # 对应Zi
#         zed2   = qele2                                  # 对应Zj
#         ev     = 1.602176565e-19                        # 对应e
#         pi     = 3.14159265358979324e0                  # 对应π
#         epsil0 = 8.854187817e-12                        # 对应ε0
#         bohrad = 0.52917721067e0                       # 波尔半径：0.529埃
#         exn    = 0.23e0                                # 幂函数的指数（常数）
#         beta   = (zed1*zed2*ev*ev)/(4.0*pi*epsil0)*1.0e10/ev
#         rs     = 0.8854*bohrad/(zed1**exn +zed2**exn)   #  对应公式a = 0.4685335/(Zi^0.23 + Zj^0.23)
#         bzb[0] = -3.19980 / rs     #  对应x = rij/a
#         bzb[1] = -0.94229 / rs     #  对应x = rij/a
#         bzb[2] = -0.40290 / rs     #  对应x = rij/a
#         bzb[3] = -0.20162 / rs     #  对应x = rij/a
#         v[i]   = beta * (0.18175 * np.exp(bzb[0] * r) +
#                          0.50986 * np.exp(bzb[1] * r) +
#                          0.28022 * np.exp(bzb[2] * r) +
#                          0.02817 * np.exp(bzb[3] * r))
#
#     if r < 1.0:
#         zed1 = qele1
#         zed2 = qele2
#         ev = 1.602176565e-19
#         pi = 3.14159265358979324e0
#         epsil0 = 8.854187817e-12
#         bohrad = 0.52917721067e0
#         exn = 0.23e0
#         beta = (zed1*zed2*ev*ev)/(4.0*pi*epsil0)*1.0e10/ev
#         rs = 0.8854*bohrad/(zed1**exn +zed2**exn)
#         bzb[0] = -3.19980/rs
#         bzb[1] = -0.94229/rs
#         bzb[2] = -0.40290/rs
#         bzb[3] = -0.20162/rs
#         rinv = 1.0/r
#         vsum[i] = beta * rinv * (0.18175 * np.exp(bzb[0] * r) +
#                                  0.50986 * np.exp(bzb[1] * r) +
#                                  0.28022 * np.exp(bzb[2] * r) +
#                                  0.02817 * np.exp(bzb[3] * r))
#
#     elif r >= 1.0 and r < 2.3:
#         vsum[i] = np.exp(b0+b1*r+b2*r**2.0+b3*r**3.0)
#
#     elif r >= 2.3:
#         vsum[i] = T_ap[0] * (T_rp[0] - r) ** 3 * HH(T_rp[0] - r) + \
#         T_ap[1] * (T_rp[1] - r) ** 3 * HH(T_rp[1] - r) + \
#         T_ap[2] * (T_rp[2] - r) ** 3 * HH(T_rp[2] - r) + \
#         T_ap[3] * (T_rp[3] - r) ** 3 * HH(T_rp[3] - r) + \
#         T_ap[4] * (T_rp[4] - r) ** 3 * HH(T_rp[4] - r) + \
#         T_ap[5] * (T_rp[5] - r) ** 3 * HH(T_rp[5] - r) + \
#         T_ap[6] * (T_rp[6] - r) ** 3 * HH(T_rp[6] - r) + \
#         T_ap[7] * (T_rp[7] - r) ** 3 * HH(T_rp[7] - r) + \
#         T_ap[8] * (T_rp[8] - r) ** 3 * HH(T_rp[8] - r) + \
#         T_ap[9] * (T_rp[9]-r) ** 3 * HH(T_rp[9] - r) - \
#  2.0 * T_ccc * (T_ao[0] * (T_ro[0] - r) ** 3 * HH(T_ro[0] - r) +
#                 T_ao[1] * (T_ro[1] - r) ** 3 * HH(T_ro[1] - r) +
#                 T_ao[2] * (T_ro[2] - r) ** 3 * HH(T_ro[2] - r) +
#                 T_ao[3] * (T_ro[3] - r) ** 3 * HH(T_ro[3] - r) +
#                 T_ao[4] * (T_ro[4] - r) ** 3 * HH(T_ro[4] - r) +
#                 T_ao[5] * (T_ro[5] - r) ** 3 * HH(T_ro[5] - r) +
#                 T_ao[6] * (T_ro[6] - r) ** 3 * HH(T_ro[6] - r) +
#                 T_ao[7] * (T_ro[7] - r) ** 3 * HH(T_ro[7] - r) +
#                 T_ao[8] * (T_ro[8] - r) ** 3 * HH(T_ro[8] - r) +
#                 T_ao[9] * (T_ro[9] - r) ** 3 * HH(T_ro[9] - r))
#
#     r_value.append(r)    # 把r存起来
#     pair_potential.append(vsum[i])  # 把对势的值存起来（没有乘以r的）
#     v[i] = r * vsum[i]
#
# np.savetxt('r_value.txt', r_value)
# np.savetxt('pair_potential_value.txt', pair_potential)   # 保存没有乘以r的对势值
#
# r_test = []
# pair_test = []
# for r, pair in zip(r_value, pair_potential):
#     if np.isnan(np.array(pair)):
#         pass
#     else:
#         r_test.append(r)
#         pair_test.append(pair)
#
# plt.plot(r_test, pair_test)
# plt.xlim(1.5, 6)
# plt.ylim(-0.5, 3)
# plt.show()
#
#
# for i in range(0, int(10000 / 5)):  # 分成5列写入到势函数文件中
#     file.write("{:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\t {:24.14E}\n".
#                format(v[5 * i], v[5 * i + 1], v[5 * i + 2], v[5 * i + 3],
#                       v[5 * i + 4]))
#
# file.close()
#
#
#
#
#
#
