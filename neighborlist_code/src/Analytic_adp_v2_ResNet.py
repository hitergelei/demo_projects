from ase import Atoms
from ase.io import read, write
from ase.build import bulk, make_supercell
from ase.neighborlist import NeighborList
from ase.calculators.eam import EAM
import matplotlib.pyplot as plt
import numpy as np
import asap3
from torch_neigh import TorchNeighborList
from adp_func import emb, emb_vec, rho, phi_AA, u11, w11

# vesin也提供了近邻列表的计算
# https://github.com/Luthaf/vesin


def ase_neigh(atoms_obj: Atoms, cutoff: float):  
    # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html
    from ase.neighborlist import neighbor_list as ase_neighbor_list
    print("\n----------------ase的neighbor_list计算结果-----------------------")
    atoms = atoms_obj
   # 实现求“ijdD”类型的邻居列表
    # cutoff = [cutoff for _ in range(len(atoms))]
    # nl = NeighborList(cutoff, skin=0.0, self_interaction=False, bothways=True)
    # nl.update(atoms)
    # indices, offsets = nl.get_neighbors(0)
    # print("indices = ", indices)
    # indices, offsets = nl.get_neighbors(1)
    # print("indices = ", indices)

    i, j, d, D = ase_neighbor_list('ijdD', atoms_obj, cutoff)
    print(len(i))
    print("i = \n", i)
    print("j = \n", j)
    print("d.shape = \n", d.shape)
    print("D.shape = \n", D.shape)


def asap3_get_neighborlist(cutoff, atoms):      
    nl = asap3.FullNeighborList(cutoff, atoms)
    pair_i_idx = []
    pair_j_idx = []
    n_diff = []
    for i in range(len(atoms)):
        indices, diff, _ = nl.get_neighbors(i)
        pair_i_idx += [i] * len(indices)               # local index of pair i
        pair_j_idx.append(indices)   # local index of pair j
        n_diff.append(diff)

    pair_j_idx = np.concatenate(pair_j_idx)
    pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
    n_diff = np.concatenate(n_diff)
    
    return pairs, n_diff


def Ta_adp_test():
    # Ta_adp = EAM(potential="./test_examples2/Ta.adp_modified.txt", cutoff=3.0, skin=0, elements=['Ta'], form='adp')
    # # Ta_adp.plot('Ta_adp_test')  # 触发 Ta_rij.txt和Ta_Φij.txt数据的本地保存
    
    atoms_obj = read("./test_examples2/dump.lammpstrj", format="lammps-dump-text")
    atoms_obj.set_chemical_symbols(['Ta' for _ in range(len(atoms_obj))])
    # Ta_adp.calculate(properties='energy', atoms=atoms_obj)

 
    # cutoff = 3  # 截断半径
    cutoff = 6.150958970000000e+00  # Ta的adp势文件中实际的截断半径
    TorchNeigh(atoms_obj, cutoff=cutoff)

    neighbor_list = TorchNeighborList(cutoff=cutoff)   # 已经设置好了，计算截断半径内的粒子对(i, j)
    # 计算近邻列表。 (i,j) = pairs表示粒子对 ； rvec = pair_diff 表示3个方向dx,dy,dz的距离； rij = pair_dist表示(i,j)之间的距离
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)   # np.linalg.norm(pair_diff, axis=1)计算得到pair_dist
    print("\n<--------u11(rij)*dx, u11(rij)*dy, u11(rij)*dz-------->\n")
    print(pair_diff*u11(pair_dist)[:, np.newaxis])    # mu_i^α  # 这里没有累加，只是单纯的相乘
    print((pair_diff*u11(pair_dist)[:, np.newaxis]).shape)  # u11(rij)*rij   # 3个方向: u11(rij)*dx, u11(rij)*dy, u11(rij)*dz
    # 测试asap3的计算结果，也是一样的
    # pairs_asap3, pair_diff_asap3 = asap3_get_neighborlist(cutoff, atoms_obj)  

    # 参考：ase.calculator.eam.py源码中的公式：  mu = np.sum((rvec * d(r)[:, np.newaxis]), axis=0) 以及
    # lammps中pair_adp.cpp中的u2 = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];  这里u2就是u(rij)，其中rij表示(i,j)之间的距离，是标量
    # mu_i_alpha = np.concatenate((pairs, (u11(pair_dist)*pair_diff[:, np.newaxis])), axis=1)
    mu_i_alpha = np.concatenate((pairs, pair_diff*u11(pair_dist)[:, np.newaxis]), axis=1) # mu_i_alpha[:, :2].astype(int)查看第1,2列为粒子对(i,j)的索引
    
    #------------for plotting----------
    rij_list = []
    phi_AA_list = []
    rho_i_list = []   # Electron Density rho(r)
    emb_ener_list = []
    _phi_AA_values = np.array([phi_AA(rij) for rij in pair_dist])
    _rho_i_values = np.array([rho(rij) for rij in pair_dist])
    _emb_ener_values = np.array([emb(rho_i) for rho_i in _rho_i_values])
    #------------for plotting----------
    
    pair_energy = 0.0     # ∑Φ(rij)
    embedding_energy = 0.0 # ∑F(rho_i)
    mu_energy = 0.0
    lam_energy = 0.0
    trace_energy = 0.0
    total_density = np.zeros(len(atoms_obj))
    # 参考：ase.calculators.eam.EAM中的函数和lammps的pair_adp.cpp代码
    mu = np.zeros([len(atoms_obj), 3])    # dx, dy, dz三个方向： ∑u11*dx, ∑u11*dy, ∑u11*dz # 注意：只统计[i,j]之间的,不需要再统计[j,i]之间的
    lam = np.zeros([len(atoms_obj), 3, 3])
    # print(pair_diff)
    # print(pair_diff.min(), pair_diff.max())
    # ase_neigh(atoms_obj=atoms_obj, cutoff=cutoff)   # TODO: 使用ase的neighbor_list计算效果是一致的。Q: 为什么会有重复统计的(i,j)对统计？比如(0,42)出现2次。
    
    #----way1: 标量输入的计算方法---
    if False:
        for idx, pair in enumerate(pairs):
            i, j = pair[0], pair[1]
            assert i != j, "原子对(i,j)的索引不能相同"
            # print(f"idx = {idx}, 原子对({i}, {j})之间的距离为{pair_dist[idx]:.4f}")
            # if pair_dist[idx] < cutoff:   
                # print(f"原子对({i}, {j})之间的距离为{pair_dist[idx]:.4f}")
                # print('i = {}, j = {}, phi_AA(rij) = {}'.format(i, j, phi_AA(pair_dist[idx])))
            # if i == 0:
            #     print(f"原子对({i}, {j})之间的距离为{pair_dist[idx]:.4f}")   

            rij = pair_dist[idx]   # 粒子对(i,j)之间的距离值rij
            rij_list.append(rij)
            phi_AA_list.append(phi_AA(rij))   
            rho_i = rho(rij)    # rho_i or ρi
            rho_i_list.append(rho(rij))   # Electron Density rho(ri) 
            pair_energy += phi_AA(rij) / 2.    # ∑Φ(rij) / 2
            
            emb_ener_list.append(emb(rho_i))
            embedding_energy += emb(rho_i)   # ∑ F(rho_i)
            dx, dy, dz = pair_diff[idx][0], pair_diff[idx][1], pair_diff[idx][2]

            # 测试用中心原子i=0或者53的情况
            if i == 0 or 53:
                # print(f'idx = {idx} ; rij = {rij}, u11(rij) = {u11(rij)}, dx = {dx}, dy = {dy} , dx = {dz}')
                # print(f'i={i}, j = {j}, u11(dx) = {u11(dx)}, u11(dy) = {u11(dy)}, u11(dz) = {u11(dz)}')
                print(f'i={i}, j = {j}, u11(dx)*dx = {u11(dx)*dx}, u11(dy)*dy = {u11(dy)*dy}, u11(dz)*dz = {u11(dz)*dz}')

    
    # way2: 支持rij为数组或者向量类型，直接计算。等价于上面的标量输入计算的方法        
    if True:
        phi_AA_values = phi_AA(pair_dist)
        rho_i_values = rho(pair_dist) 
        emb_ener_values = emb_vec(rho_i_values)
        # print(pair_dist)
        for atom_i in range(len(atoms_obj)):
            # 例如：选择atom_idx索引为0的粒子对
            idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i
            mu_arr = mu_i_alpha[idx_mask][:, 2:5]   # 对应x, y, z三个方向的mu_i^α (α=0,1,2分别表示x, y, z方向)
            mu[atom_i] = np.sum(mu_arr, axis=0, keepdims=True)  # keepdims可不加？结果也一样
        
        print("mu = \n", mu)
        print("mu.shape = ", mu.shape)



# TODO: 有待于实现和ase中类似的计算方法
def angular_forces(self, mu_i, mu, lam_i, lam, r, rvec, form1, form2):
    # calculate the extra components for the adp forces
    # rvec are the relative positions to atom i
    psi = np.zeros(mu.shape)
    for gamma in range(3):
        term1 = (mu_i[gamma] - mu[:, gamma]) * self.d[form1][form2](r)

        term2 = np.sum((mu_i - mu) *
                        self.d_d[form1][form2](r)[:, np.newaxis] *
                        (rvec * rvec[:, gamma][:, np.newaxis] /
                        r[:, np.newaxis]), axis=1)

        term3 = 2 * np.sum((lam_i[:, gamma] + lam[:, :, gamma]) *
                            rvec * self.q[form1][form2](r)[:, np.newaxis],
                            axis=1)
        term4 = 0.0
        for alpha in range(3):
            for beta in range(3):
                rs = rvec[:, alpha] * rvec[:, beta] * rvec[:, gamma]
                term4 += ((lam_i[alpha, beta] + lam[:, alpha, beta]) *
                            self.d_q[form1][form2](r) * rs) / r

        term5 = ((lam_i.trace() + lam.trace(axis1=1, axis2=2)) *
                    (self.d_q[form1][form2](r) * r +
                    2 * self.q[form1][form2](r)) * rvec[:, gamma]) / 3.

        # the minus for term5 is a correction on the adp
        # formulation given in the 2005 Mishin Paper and is posted
        # on the NIST website with the AlH potential
        psi[:, gamma] = term1 + term2 + term3 + term4 - term5

    return np.sum(psi, axis=0)




def Mo_adp_test():
    Mo_adp = EAM(potential="./test_Mo_Ta_adp/Mo.adp", cutoff=6.5000000000000000e+00, skin=0, elements=['Mo'], form='adp')
    # # Mo_adp.plot('Mo_adp_test')  # 触发 Mo_rij.txt和Mo_Φij.txt数据的本地保存
    
    atoms_obj = read("Mo_structs_0.dat", format="lammps-data")
    atoms_obj.set_chemical_symbols(['Mo' for _ in range(len(atoms_obj))])
    Mo_adp.calculate(properties='energy', atoms=atoms_obj)   # 测试例子：调试ase的源代码，查看我的计算结果是否正确

 
    # cutoff = 3  # 截断半径
    cutoff = 6.5000000000000000e+00  # Mo的adp势文件中实际的截断半径
    TorchNeigh(atoms_obj, cutoff=cutoff)

    neighbor_list = TorchNeighborList(cutoff=cutoff)   # 已经设置好了，计算截断半径内的粒子对(i, j)
    # 计算近邻列表
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)
    # 测试asap3的计算结果，也是一样的
    # pairs_asap3, pair_diff_asap3 = asap3_get_neighborlist(cutoff, atoms_obj)  

    print("\n<--------u11(rij)*dx, u11(rij)*dy, u11(rij)*dz-------->\n")

    # TODO: 目前mu_i_alpha和lam_i_alpha_beta的设计，只支持纯元素
    # 将 u11 扩展为形状 (3456, 1)
    print(pair_diff*u11(pair_dist)[:, np.newaxis])    # mu_i^α  # 这里没有累加，只是单纯的相乘。 或者这样写：u11(pair_dist)[:, np.newaxis] * pair_diff
    print((pair_diff*u11(pair_dist)[:, np.newaxis]).shape)  # u11(rij)*rij   # 3个方向: u11(rij)*dx, u11(rij)*dy, u11(rij)*dz
    mu_i_alpha = np.concatenate((pairs, pair_diff*u11(pair_dist)[:, np.newaxis]), axis=1) # mu_i_alpha[:, :2].astype(int)查看第1,2列为粒子对(i,j)的索引
    
    
    print("\n<--------w11(rij)*rij^αβ (α=0,1,2; β=0,1,2；代表x,y,z这3个方向)-------->\n")
    # 计算rij^αβ所有组合(对于每个粒子对(i, j)，可以得到9个分量。 resut这里看成是rij^α与rij^β的两两互乘，组成的数组
    result = np.einsum('ij,ik->ijk', pair_diff, pair_diff)  # 每个粒子对，都有一个(3x3)的矩阵（该矩阵类似于协方差矩阵一样）
    print('result.shape = ', result.shape)  # (N, 3, 3)  这里为(i, j)粒子对的个数
    
    #---------------------------------写法(1)：只提取6个分量，lammps的写法---------------------------------
    # # 从（N, 3, 3)数组中提取所需元素：考虑到对称性（如dx*dy = dy*dx)，所以只需要dx*dx, dy*dy, dz*dz和dx*dy, dx*dz, dy*dz即可)
    # dx_dx = result[:, 0, 0]
    # dy_dy = result[:, 1, 1]
    # dz_dz = result[:, 2, 2]
    # dy_dz = result[:, 1, 2]
    # dx_dz = result[:, 0, 2]
    # dx_dy = result[:, 0, 1]
    # # 将结果组合成一个二维数组（注意：列拼接的顺序可以随意改的，我这里根据的是lammps中adp的lambda的顺序, 而不是ase.calculator.eam中lam的顺序）
    # rij_alpha_rij_beta = np.column_stack([dx_dx, dy_dy, dz_dz, dy_dz, dx_dz, dx_dy])  # 注：rij^α*rij^β (包含了α=β的情况)
    # print(w11(pair_dist)[:, np.newaxis] * rij_alpha_rij_beta)  # lam_i^αβ (并且这里包含了α=β的情况)  # 这里没有累加，只是单纯的相乘
    # lam_i_alpha_beta = np.concatenate((pairs, w11(pair_dist)[:, np.newaxis] * rij_alpha_rij_beta), axis=1) 
    # print("lam_i_alpha_beta = ", lam_i_alpha_beta)
    #---------------------------------------------------------------------------------------------------
    
    #------------for plotting----------
    rij_list = []
    phi_AA_list = []  # Φ(rij)
    rho_i_list = []   # Electron Density rho(r) 
    emb_ener_list = [] # F(rho_i)
    _phi_AA_values = np.array([phi_AA(rij) for rij in pair_dist])
    _rho_i_values = np.array([rho(rij) for rij in pair_dist])
    _emb_ener_values = np.array([emb(rho_i) for rho_i in _rho_i_values])
    #------------for plotting----------
    
    cfg_energy = 0.0
    embedding_energy = 0.0  # 公式的第1项：∑F(rho_i)
    embed_ener_list = []
    pair_energy = 0.0       # 公式的第2项：∑Φ(rij) / 2
    mu_energy = 0.0         # 公式的第3项：∑mu_i^α^2 / 2
    lam_energy = 0.0        # 公式的第4项：∑lam_i^αβ^2 / 2
    trace_energy = 0.0      # 公式的第5项：-∑νi^2 / 6
    total_density = np.zeros(len(atoms_obj))
    # 参考：ase.calculators.eam.EAM中的函数和lammps的pair_adp.cpp代码
    mu = np.zeros([len(atoms_obj), 3])    # dx, dy, dz三个方向： ∑u11*dx, ∑u11*dy, ∑u11*dz # 注意：只统计[i,j]之间的,不需要再统计[j,i]之间的
    _lam = np.zeros([len(atoms_obj), 1, 6])  #  在写法1中：对应lammps代码中∑lam_i^αβ^2 / 2中dx*dx, dy*dy, dz*dz, dy*dz, dx*dz, dx*dy的写法
    lam_maxtri_9 = np.zeros([len(atoms_obj), 3, 3])  #  在写法2中：对应lammps代码中∑lam_i^αβ^2 / 2
    cfg_forces = np.zeros((len(atoms_obj), 3))

    
    # print(pair_diff)
    # print(pair_diff.min(), pair_diff.max())
    # ase_neigh(atoms_obj=atoms_obj, cutoff=cutoff)   # TODO: 使用ase的neighbor_list计算效果是一致的。Q: 为什么会有重复统计的(i,j)对统计？比如(0,42)出现2次。
    
    #----way1: 标量输入的计算方法---
    if False:
        for idx, pair in enumerate(pairs):
            i, j = pair[0], pair[1]
            assert i != j, "原子对(i,j)的索引不能相同"
            # print(f"idx = {idx}, 原子对({i}, {j})之间的距离为{pair_dist[idx]:.4f}")
            # if pair_dist[idx] < cutoff:   
                # print(f"原子对({i}, {j})之间的距离为{pair_dist[idx]:.4f}")
                # print('i = {}, j = {}, phi_AA(rij) = {}'.format(i, j, phi_AA(pair_dist[idx])))
            # if i == 0:
            #     print(f"原子对({i}, {j})之间的距离为{pair_dist[idx]:.4f}")   

            rij = pair_dist[idx]   # 粒子对(i,j)之间的距离值rij
            rij_list.append(rij)
            phi_AA_list.append(phi_AA(rij))   
            rho_i = rho(rij)    # rho_i or ρi
            rho_i_list.append(rho(rij))   # Electron Density rho(ri) 
            pair_energy += phi_AA(rij) / 2.    # ∑Φ(rij) / 2
            
            emb_ener_list.append(emb(rho_i))
            embedding_energy += emb(rho_i)   # ∑ F(rho_i)
            dx, dy, dz = pair_diff[idx][0], pair_diff[idx][1], pair_diff[idx][2]

            # 测试用中心原子i=0或者53的情况
            if i == 0 or 53:
                # print(f'idx = {idx} ; rij = {rij}, u11(rij) = {u11(rij)}, dx = {dx}, dy = {dy} , dx = {dz}')
                # print(f'i={i}, j = {j}, u11(dx) = {u11(dx)}, u11(dy) = {u11(dy)}, u11(dz) = {u11(dz)}')
                print(f'i={i}, j = {j}, u11(dx)*dx = {u11(dx)*dx}, u11(dy)*dy = {u11(dy)*dy}, u11(dz)*dz = {u11(dz)*dz}')

    
    # way2: 支持rij为数组或者向量类型，直接计算。等价于上面的标量输入计算的方法        
    if True:
        for atom_i in range(len(atoms_obj)):
            # 例如：选择atom_idx索引为0的粒子对
            idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i
            
            phi_AA_values = phi_AA(pair_dist[idx_mask])
            pair_energy += np.sum(phi_AA_values) / 2.    # 公式的第2项：∑Φ(rij) / 2
            rho_i_values = rho(pair_dist[idx_mask])  #  ρ(rij)  对应ase中的self.electron_density[j_index](r[nearest][use])
            density = np.sum(rho_i_values, axis=0)   # ρ(i)  选择对应原子的rho_i
            total_density[atom_i] = np.sum(rho(pair_dist[idx_mask]), axis=0)   # 测试用： 每个中心原子i对应的ρ(i)的结果。
            emb_ener_values = emb_vec(density)  # emb_vec表示向量化的emb函数: F(rho_i)
            embedding_energy += np.sum(emb_ener_values)   # 公式的第1项：∑F(rho_i)
            embed_ener_list.append(emb_ener_values.tolist())   # 用于测试ase的结果: 每个中心原子i对应的F(rho_i))
            #------------

            
            mu_arr = mu_i_alpha[idx_mask][:, 2:5]   # 对应x, y, z三个方向的mu_i^α (α=0,1,2分别表示x, y, z方向)
            mu[atom_i] = np.sum(mu_arr, axis=0)  # keepdims可不加？结果也一样 # np.sum(mu_arr, axis=0, keepdims=True)
            
            # ---------------写法(1): 用于6分量的写法--------未实现成功!------------
            # TODO: 可以参考：https://gitlab.com/atomicrex/atomicrex/-/blob/master/src/arexlib/potentials/AngularDependentPotential.cpp?ref_type=heads#L35
            # # lam_i_alpha_beta为（N, 8）数组，包含了对角线元素. 前2列对应粒子对(i, j)，后6列对应lammps代码中dx*dx, dy*dy, dz*dz, dy*dz, dx*dz, dx*dy的写法

            # lam_arr = lam_i_alpha_beta[idx_mask]   # lam_arr[:, :2].astype(int) # 第1,2列为粒子对(i,j)的索引
            # lam_arr_alpha_alpha = lam_arr[:, 2:5]  # 对应dx*dx, dy*dy, dz*dz  公式(6)中的niu_i （未累加时）
            # lam_arr_alpha_beta = lam_arr[:, 5:8]   # 对应dy*dz, dx*dz, dx*dy. 公式(5)中的lam_i^αβ （未累加时）
            # _lam[atom_i][:, 0:3] = 0.5*np.sum(lam_arr_alpha_alpha*lam_arr_alpha_alpha, axis=0)  # (N, 3) # np.sum(np.square(lam_arr_alpha_alpha), axis=0)
            # _lam[atom_i][:, 3:6] = 1.0*np.sum(lam_arr_alpha_beta*lam_arr_alpha_beta, axis=0)    # (N, 3)
            # print("00000000-------------00000000-------------")
            
            # ------------------------------写法(2): 用于9分量的写法--------------------------------------
            # # lam_arr为（N, 3, 3）数组，包含了对角线元素. 对应x, y, z三个方向的lam_i^αβ 
            rvec= pair_diff[idx_mask]
            # 初始化 lam 矩阵
            lam = np.zeros([rvec.shape[0], 3, 3])
            r = pair_dist[idx_mask]
            qr = w11(r)
            # for alpha in range(3):
            #     for beta in range(3):
            #         lam[:, alpha, beta] += qr * rvec[:, alpha] * rvec[:, beta]
            
            # 使用 np.einsum 替换alpha和beta这2个for循环
            lam = np.einsum('i,ij,ik->ijk', qr, rvec, rvec)
            lam_maxtri_9[atom_i] = np.sum(lam, axis=0)  # 9个分量的矩阵. 对应ase中的self.lam[i]
            print(lam_maxtri_9[0])
            print(lam_maxtri_9.shape)
            # #-----------------------------------------------------------------------------------------

        print("\n------> embed_ener_list(即，每个中心原子i对应的F(rho_i)) ------ \n", embed_ener_list)  
        print("\n------> total_density(即，每个中心原子i对应的ρ(i))值----- \n", total_density)   
        print("mu = \n", mu)
        print("mu.shape = ", mu.shape)
        # 如果要自动微分求导，请确保mu是一个torch.Tensor类型的对象而不是普通的numpy数组  #torch.sum(mu ** 2) / 2.
        mu_energy += np.sum(mu ** 2) / 2.   # 公式的第3项：∑mu_i^α^2 / 2


        lam_energy += np.sum(lam_maxtri_9 ** 2) / 2.     # 公式的第4项：∑lam_i^αβ^2 / 2
        for i in range(len(atoms_obj)):  # this is the atom to be embedded
            trace_energy -= np.sum(lam_maxtri_9[i].trace() ** 2) / 6.     # 公式中的第5项： -∑νi^2 / 6

        
        cfg_energy =  pair_energy + embedding_energy + mu_energy + lam_energy + trace_energy
        print("cfg_energy = ", cfg_energy)




    # plt.scatter(rij_list, phi_AA_list, s=2, c='red')
    # plt.scatter(pair_dist, _phi_AA_values, s=2, c='blue')
    # plt.scatter(pair_dist, _phi_AA_values, s=2, c='green')
    # plt.xlim(0, 6)
    # plt.ylim(-2, 2)
    # plt.legend(['scalar1', 'scalar2', 'vector'])
    # plt.savefig("Mo_phi_AA.jpg")

    # plt.scatter(rij_list, rho_i_list, s=2, c='red')
    # plt.scatter(pair_dist, _rho_i_values, s=2, c='blue')
    # plt.scatter(pair_dist, rho_i_values, s=2, c='green')
    # plt.xlim(1, 7)
    # plt.ylim(-1, 10)
    # plt.legend(['scalar1', 'scalar2', 'vector'])
    # plt.savefig("Mo_rho(r).jpg")

    # plt.scatter(rho_i_list, emb_ener_list, s=2, c='red')
    # plt.scatter(_rho_i_values, _emb_ener_values, s=2, c='blue')
    # plt.scatter(rho_i_values, emb_ener_values, s=2, c='green')
    # plt.legend(['scalar1', 'scalar2', 'vector'])
    # plt.savefig("Mo_emb_ener.jpg")






    # ------rij与Φij的关系验证-------即phi_AA
    # _rij = np.loadtxt("Mo_rij.txt")
    # _phi_rij = np.loadtxt("Mo_Φij.txt")
    # plt.xlim(0, 6)
    # plt.ylim(-2, 2)
    # plt.scatter(rij_list, phi_AA_list, marker='o')
    # plt.plot(_rij, _phi_rij, color='red')
    # plt.savefig("./Mo_rij_Φij.jpg")
    
    # ------rij与rho_i的关系验证-------
    # _rij = np.loadtxt("Mo_rij.txt")
    # _rho_i_list = np.loadtxt("Mo_rho_i.txt")
    # plt.scatter(rij_list, rho_i_list, marker='o', s=2)
    # plt.plot(_rij, _rho_i_list, color='red')
    # plt.xlim(1, 7)
    # plt.ylim(-1, 10)
    # plt.savefig("./Mo_rij_rho_i.jpg")

def TorchNeigh(atoms_obj: Atoms, cutoff: float):
    print("\n----------------TorchNeighborList方法")
    # 实例化邻居列表类
    cutoff = cutoff  # 设置截断距离
    neighbor_list = TorchNeighborList(cutoff=cutoff)

    # 计算近邻列表
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)
    
    print("len(pairs) = ", len(pairs))
    # pairs包含原子对的索引，pair_diff是原子对之间的向量差，pair_dist是原子对之间的距离
    print("Pairs:\n", pairs)
    print("Pair differences形状：", pair_diff.shape)
    # print("Pair differences:\n", pair_diff)
    # print("Pair distances:\n", pair_dist)
    print("Pair distances形状: ", pair_dist.shape)


            

if __name__ == "__main__":
    Mo_adp_test()
    # Ta_adp_test()
    # pass









"""
In [17]: # 假设以下变量已经定义
    ...: atoms_obj = [0, 1, 2, 3, 4]  # 示例原子对象列表
    ...: mu_i_alpha = np.array([
    ...:     [0, 0, 1.0, 2.0, 3.0],
    ...:     [0, 0, 4.0, 5.0, 6.0],
    ...:     [0, 0, 7.0, 8.0, 9.0],
    ...:     [1, 0, 1.5, 2.5, 3.5],
    ...:     [1, 0, 4.5, 5.5, 6.5],
    ...:     [1, 0, 7.5, 8.5, 9.5],
    ...:     [2, 0, 1.0, 2.0, 3.0],
    ...:     [2, 0, 4.0, 5.0, 6.0],
    ...:     [2, 0, 7.0, 8.0, 9.0]
    ...: ])
    ...:
    ...: # 初始化 mu 数组
    ...: mu = np.zeros((len(atoms_obj), 3))
    ...:
    ...: for atom_i in range(len(atoms_obj)):
    ...:     # 选择对应原子的索引
    ...:     idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i
    ...:     mu_arr = mu_i_alpha[idx_mask][:, 2:5]  # 对应 x, y, z 三个方向的 mu_i^α (α=0,1,2 分别表示 x, y, z 方向)
    ...:     mu[atom_i] = np.sum(mu_arr, axis=0, keepdims=True)  # keepdims 可以不加，但保留维度更方便后续操作
    ...:
    ...: print("mu = \n", mu)
    ...: print("mu.shape = ", mu.shape)
mu =
 [[12.  15.  18. ]
 [13.5 16.5 19.5]
 [12.  15.  18. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]]
mu.shape =  (5, 3)
"""


#----------------------残差网络------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU(inplace=True)  # 确保这里定义了ReLU激活函数

    def forward(self, x):
        return x + self.relu(self.linear(x))

# 定义残差网络模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)  # 在这里定义ReLU激活函数
        self.linear1 = nn.Linear(1, 50)  # 假设输入特征维度为1
        self.block1 = nn.Sequential(
            block(50),
            block(50),
            block(50)
        )
        self.block2 = nn.Sequential(
            block(50),
            block(50),
            block(50)
        )
        self.block3 = nn.Sequential(
            block(50),
            block(50),
            block(50)
        )
        self.linear2 = nn.Linear(50, 1)  # 假设输出特征维度为1

    def forward(self, x):
        x = self.relu(self.linear1(x))  # 使用定义好的ReLU激活函数
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.linear2(x)

# 实例化残差块和模型
block = ResidualBlock
model = ResNet(block, 3)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 准备训练数据
x_train = torch.randn(100, 1)  # 随机生成的输入数据
y_train = x_train.pow(2) + 0.1 * torch.randn(100, 1)  # 真实函数 y = x^2，加上噪声

# 训练模型
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    predicted = model(x_train)
    loss = criterion(predicted, y_train)
    print(f'Test Loss: {loss.item():.4f}')