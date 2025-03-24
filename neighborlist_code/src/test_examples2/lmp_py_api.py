from mpi4py import MPI
import sys
from lammps import lammps
import math
import numpy as np  

"""并行运行：
mpirun -n 4 python lmp_py_api.py 
"""




# 获取Python版本
python_version = sys.version
print(f"Python version: {python_version}")
# # 获取Python版本
# python_version = "Python " + sys.version.split()[0]
# print("Python version:", python_version)

# 创建 LAMMPS 实例
lmp = lammps()

# 设置输入文件和日志文件
input_file = "in.file"


# 运行 LAMMPS 命令
lmp.file(input_file)


# 获取并打印 LAMMPS 版本
print("LAMMPS Version:", lmp.version())

# # 获取 MPI 通信器
# comm = MPI.COMM_WORLD
# print("Proc %d out of %d procs" % (comm.Get_rank(), comm.Get_size()))

cell_info = lmp.extract_box()
natoms = lmp.get_natoms()
print("natoms = ", natoms)


# https://docs.lammps.org/Python_atoms.html
# https://github.com/lammps/lammps/tree/develop/python/examples
nlocal = lmp.extract_global("nlocal")
x = lmp.numpy.extract_atom("x")    # extract a per-atom quantity as numpy array 。这里是各个原子的3个方向坐标
# x = lmp.extract_atom("x")        # extract a per-atom quantity

for i in range(nlocal):
   print("(x,y,z) = (", x[i][0], x[i][1], x[i][2], ")")
        


# https://docs.lammps.org/Python_neighbor.html
# look up the neighbor list
# 注:如果把Ta.adp.txt的势函数文件中的截断半径6.150958970000000e+00改成3.0，结果跟我们torch_neighborlist_code.py测试的例子是一样的
nlidx = lmp.find_pair_neighlist('adp')   
nl = lmp.numpy.get_neighlist(nlidx)

tags = lmp.extract_atom('id')   # lmp.numpy.extract_atom('id')
print("half neighbor list with {} entries".format(nl.size))
# print neighbor list contents
for i in range(0,nl.size):
    idx, nlist  = nl.get(i)
    print("\natom {} with ID {} has {} neighbors:".format(idx,tags[idx],nlist.size))
    if nlist.size > 0:
        # np.nditer 用于遍历邻居列表 nlist 中的元素
        for n in np.nditer(nlist):
            print("  atom {} with ID {}".format(n,tags[n]))


print("---------->判断粒子对之间的距离，如果小于3.0，则输出")

# 获取模拟箱的尺寸
boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()


# 计算周期性边界条件下的距离
for i in range(0, nl.size):
    idx, nlist = nl.get(i)
    
    print("\natom {} with ID {} has {} neighbors:".format(idx, tags[idx], nlist.size))
    
    if nlist.size > 0:
        for n in np.nditer(nlist):
            # 检查 n 是否在有效范围内
            if n >= len(x):
                print(f"Warning: n {n} is out of bounds for x array of size {len(x)}")
                continue

            print("  atom {} with ID {}".format(n, tags[n]))
            
            # 相对位置向量
            dx = x[idx] - x[n]
            # 考虑周期边界条件： xy, xz, yz 表示斜率（倾斜因子），用于处理非正交的模拟箱。
            # 这段代码的目的是确保 dx 在周期性边界内，即当 dx 超出模拟箱边界时，将其调整到正确的周期性位置
            # 计算了 dx[0] 加上 xy / 2 后相对于箱长 (boxhi[0] - boxlo[0]) 的比例。np.rint 函数将浮点数四舍五入到最接近的整数，得到 dx[0] 需要调整的箱长倍数。
            """
            总结
            这段代码的主要作用是确保两个原子之间的相对位置向量 dx 在模拟箱的周期性边界内。具体步骤如下：
            1. 计算相对位置向量 dx。
            2. 处理每个维度的周期性边界条件：
               (1) 计算偏移量：(dx[0] + xy / 2) / (boxhi[0] - boxlo[0])
               (2) 取最近整数倍的箱长: np.rint操作
               (3) 调整 dx，使其落入周期性边界内。
            这样可以保证在非正交的模拟箱中，两个原子之间的相对位置正确地反映周期性边界条件。
            """
            dx[0] -= np.rint((dx[0] + xy / 2) / (boxhi[0] - boxlo[0])) * (boxhi[0] - boxlo[0])  
            dx[1] -= np.rint((dx[1] + xz / 2) / (boxhi[1] - boxlo[1])) * (boxhi[1] - boxlo[1])
            dx[2] -= np.rint((dx[2] + yz / 2) / (boxhi[2] - boxlo[2])) * (boxhi[2] - boxlo[2])
            r = np.sqrt(np.sum(dx**2))
            
            if r < 3.0:
                print("=======>>> atom {} with ID {} is at distance {}".format(n, tags[n], r))


# 显式关闭 LAMMPS 实例（如果需要）
lmp.close()

