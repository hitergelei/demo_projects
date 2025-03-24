import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

potential_file_old = "Mg.eam.fs"        # 原始Fortran生成的势函数列表文件，可能存在无穷大的值
# potential_file_old = "Mg_20190916.eam.fs" # Mg原始Fortran生成的势函数列表文件，可能存在无穷大的值
potential_file_new = "../Mg_2021test_chj.eam.fs"  # 用Python生成的势函数列表文件，已经过滤了无穷大的值     -- 注：12-3号生成的
r_value = np.loadtxt('../r_value.txt')

old_data = np.loadtxt(potential_file_old, skiprows=6)   # 跳过前6行，直接从第7行开始读取数据
new_data = np.loadtxt(potential_file_new, skiprows=6)    # 跳过前6行，直接从第7行开始读取数据

# points = eval(input("points点个数："))      # 如10000
n = eval(input("写入势函数文件的类型数："))   # 如3种

assert any([type(old_data) == np.ndarray, type(new_data) == np.ndarray])
assert old_data.shape == new_data.shape   # 例如shape值为(6000,5)

old_data_list = [None for i in range(n)]
idx = int(new_data.shape[0] / n)  # 6000/3=2000
for k in range(n):   # k=0,1,2
    old_data_list[k] = old_data[idx*k: idx*(k+1)]  # old_data_list = [old_data[0:2000], old_data[2000:4000], old_data[4000:6000]]


new_data_list = [None for i in range(n)]
for k in range(n):   # k=0,1,2
    new_data_list[k] = new_data[idx*k: idx*(k+1)]

# 取索引-1是因为最后一个是代表势函数的值
old_pair_r = np.array([i/j for i, j in zip(old_data_list[-1].reshape(-1,1),r_value.reshape(-1,1))])  # 除以r了
new_pair_r = np.array([i/j for i, j in zip(new_data_list[-1].reshape(-1,1),r_value.reshape(-1,1))])  # 除以r了


plt.plot(r_value, old_pair_r.reshape(-1,1), color='red')
plt.plot(r_value, new_pair_r.reshape(-1,1), color='blue')   # 真实数据
plt.xlim(1.5, 6)
plt.ylim(-0.5, 3)
plt.xlabel('r/$r$')
plt.ylabel('pair potential')
plt.title('pair potential - r using Fortran vs Python')
plt.legend(['Fortran_pair', 'Python_pair'], loc = 'upper right')
plt.show()



#-----------------读取用Python跑的数据----------------------------
r_value = np.loadtxt('../r_value.txt')
pair_potential_value = np.loadtxt('../pair_potential_value.txt')
r_filter = []
pair_filter = []
for r, pair in zip(r_value, pair_potential_value): # 对一些存在nan值的进行过滤
    if np.isnan(np.array(pair)):
        pass
    else:
        r_filter.append(r)
        pair_filter.append(pair)

plt.plot(r_value[:], pair_filter[:], color='green')   #  纵坐标是除以r的
plt.xlim(1.5, 6)
plt.ylim(-0.5, 3)
plt.title('pair potential - r using Python')   # 针对对势的值存在nan值的标准处理
plt.show()


print('>>>>>>>进行每阶段的新旧数据比对...')
print('------------new')
for i in range(n):
    print('new%s' %i, np.sum(new_data_list[i].reshape(-1,1)))
print('------------old')
for i in range(n):
    print('old%s' %i, np.sum(old_data_list[i].reshape(-1,1)))

print('比对结束！')

# print('chj1:', np.sum(chj_data_raw1.reshape(-1,1)))
# print('chj2:', np.sum(chj_data_raw2.reshape(-1,1)[1:]))
# print('chj3:', np.sum(chj_data_raw3.reshape(-1,1)))
# print('chj4:', np.sum(chj_data_raw4.reshape(-1,1)[1:]))
# print('chj5:', np.sum(chj_data_raw5.reshape(-1,1)[1:]))
# print('------------------------------------')
# print('lipan1:', np.sum(lipan_data_raw1.reshape(-1,1)))
# print('lipan2:', np.sum(lipan_data_raw2.reshape(-1,1)[1:]))
# print('lipan3:', np.sum(lipan_data_raw3.reshape(-1,1)))
# print('lipan4:', np.sum(lipan_data_raw4.reshape(-1,1)[1:]))
# print('lipan5:', np.sum(lipan_data_raw5.reshape(-1,1)[1:]))


