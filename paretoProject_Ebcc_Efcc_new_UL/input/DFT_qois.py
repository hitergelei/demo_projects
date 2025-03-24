
# ++++++++++++++++++++++++++++++++++++++++++++++++ 手动配置DTF计算的qois名字和值 +++++++++++++++++++++++++++++++++++++++

element_name = "Mg"   # 指定计算的元素对象为Zn

# 手动设置qoi名字和对应的DFT计算值（参考值）
# qoi_name_list = ["B", "C11", "C12", "C13", "C33", "C44", "Ec"]  # TO DO 2020-12-16: 把晶格常数a0也要当做是一个目标量
# qoi_value_list = [80, 179, 38, 55, 69, 46, -1.3]

#---------优化3个目标量------------
# qoi_name_list = ["deta_E_HB", "deta_E_HF", "deta_E_vf"]  # HCP和BCC结构能量差、HCP和FCC结构能量差、空位形成能
# qoi_value_list = [0.030, 0.031, 0.54]

#---------优化10个目标量------------
qoi_name_list = ["a", "Ec", "C11", "C12", "C44", "C33", "C13", "deta_E_HB", "deta_E_HF", "deta_E_vf"]
qoi_value_list = [3.209, -1.51, 63.5, 25.9, 18.4, 66.5, 21.7, 0.031, 0.026, 0.735]  # 注：deta_E_vf这里算0.58和0.89的平均值
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from collections import OrderedDict
DFT_qois_dict = OrderedDict()
for k, v in zip(qoi_name_list, qoi_value_list):
    DFT_qois_dict[k] = v

print(DFT_qois_dict)