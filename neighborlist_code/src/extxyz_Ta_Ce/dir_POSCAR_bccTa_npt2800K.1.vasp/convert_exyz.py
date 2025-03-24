import ase
from ase.io import read, write
import os

# https://wiki.fysik.dtu.dk/ase/tips.html
# /home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta/vasp/dir_POSCAR_0.900/OUTCAR
# exyz_Ta = read('/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta/vasp/dir_POSCAR_0.900/OUTCAR', format='vasp-out')
# print(exyz_Ta)

path = '/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta_Ce/vasp'
idx_list = ['0.900', '1.000', '1.100']

# # 读取每个 OUTCAR 文件并存储为 ASE Atoms 对象
structures_list = []
# for idx in idx_list:
#     outcar_paths = os.path.join(path, f'dir_POSCAR_{idx}/OUTCAR')
#     st = read(outcar_paths, format='vasp-out')
#     # print(st)
#     structures_list.append(st)

#--------------Ta表面的结构信息----------------
# st11 = read('/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta_Ce/dir_POSCAR_SLAB100_Lay3_10_SC221.cell0.05.coord0.05_idx0.vasp/OUTCAR', format='vasp-out')
# structures_list.append(st11)
# outpath = '/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta_Ce/Ta_unitcell_and_surface.extxyz'
# write(outpath, structures_list, format='extxyz', parallel=True)



# ---------------Ta熔体的结构信息----------------
st2 = read('/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta_Ce/dir_POSCAR_bccTa_npt2800K.1.vasp/OUTCAR', format='vasp-out')
structures_list.append(st2)

#---------------Ta扩胞微扰的结构信息----------------
# st22 = read('/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta_Ce/dir_POSCAR.cell0.05.coord0.1_idx1.vasp/OUTCAR', format='vasp-out')
# structures_list.append(st22)

# #---------------Ta100_Ce固液界面的结构信息----------------   
# st3 = read('/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta_Ce/Ta100_liquidCe_MLFF_NPT_2000K_dir_POSCAR102.vasp/OUTCAR', format='vasp-out')
# structures_list.append(st3)

outpath = '/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Ta_Ce/Ta_melt.extxyz'
write(outpath, structures_list, format='extxyz', parallel=True)



