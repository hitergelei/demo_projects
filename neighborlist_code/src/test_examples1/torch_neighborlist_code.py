from torch import nn
import torch
from typing import Dict, Tuple
from ase import Atoms

def wrap_positions(positions: torch.Tensor, cell: torch.Tensor, eps: float=1e-7) -> torch.Tensor:
    """Wrap positions into the unit cell"""
    # wrap atoms outside of the box
    scaled_pos = positions @ torch.linalg.inv(cell) + eps
    scaled_pos %= 1.0
    scaled_pos -= eps
    return scaled_pos @ cell

class TorchNeighborList(nn.Module):
    """Neighbor list implemented via PyTorch. The correctness is verified by comparing results to ASE and asap3.
    This class enables the direct calculation of gradients dE/dR.
    
    The speed of this neighbor list algorithm is faster than ase while being about 2 times slower than asap3 if use a sing CPU.
    If use a GPU, it is usually slightly faster than asap3.
    
    Note that current GNN implementations used a lot `atomicAdd` operations, which can result in non-deterministic behavior in the model.
    Model predictions (forces) will be erroneous if using a neighbor list algorithm that different with model training.
    """
    def __init__(
        self,
        cutoff: float=5.0,
        wrap_atoms: bool=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        disp_mat = torch.cartesian_prod(
            torch.arange(-1, 2),
            torch.arange(-1, 2),
            torch.arange(-1, 2),
        )
        self.cutoff = cutoff
        self.wrap_atoms = wrap_atoms
        self.register_buffer('disp_mat', disp_mat, persistent=False)
    
    def forward(self, atoms: Atoms) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.from_numpy(atoms.get_positions())
        cell = torch.from_numpy(atoms.get_cell()[:])
        return self._build_neighbor_list(positions, cell)

    def _build_neighbor_list(self, positions: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate padding size. It is useful for all kinds of cells
        wrapped_pos = wrap_positions(positions, cell) if self.wrap_atoms else positions
        norm_a = cell[1].cross(cell[2], dim=-1).norm()
        norm_b = cell[2].cross(cell[0], dim=-1).norm()
        norm_c = cell[0].cross(cell[1], dim=-1).norm()
        volume = torch.sum(cell[0] * cell[1].cross(cell[2], dim=-1))

        # get padding size and padding matrix to generate padded atoms. Use minimal image convention
        padding_a = torch.ceil(self.cutoff * norm_a / volume).long()
        padding_b = torch.ceil(self.cutoff * norm_b / volume).long()
        padding_c = torch.ceil(self.cutoff * norm_c / volume).long()

        padding_mat = torch.cartesian_prod(
            torch.arange(-padding_a, padding_a+1, device=padding_a.device),
            torch.arange(-padding_b, padding_b+1, device=padding_a.device),
            torch.arange(-padding_c, padding_c+1, device=padding_a.device),
        ).to(cell.dtype)
        padding_size = (2 * padding_a + 1) * (2 * padding_b + 1) * (2 * padding_c + 1)

        # padding, calculating cell numbers and shapes
        padded_pos = (wrapped_pos.unsqueeze(1) + padding_mat @ cell).view(-1, 3)
        padded_cpos = torch.floor(padded_pos / self.cutoff).long()
        corner = torch.min(padded_cpos, dim=0)[0]                 # the cell at the corner
        padded_cpos -= corner
        c_pos_shap = torch.max(padded_cpos, dim=0)[0] + 1         # c_pos starts from 0
        num_cells = int(torch.prod(c_pos_shap).item())
        count_vec = torch.ones_like(c_pos_shap)
        count_vec[0] = c_pos_shap[1] * c_pos_shap[2]
        count_vec[1] = c_pos_shap[2]
    
        padded_cind = torch.sum(padded_cpos * count_vec, dim=1)
        padded_gind = torch.arange(padded_cind.shape[0], device=count_vec.device) + 1                                 # global index of padded atoms, starts from 1
        padded_rind = torch.arange(positions.shape[0], device=count_vec.device).repeat_interleave(padding_size)                  # local index of padded atoms in the unit cell

        # atom cell position and index
        atom_cpos = torch.floor(wrapped_pos / self.cutoff).long() - corner
        atom_cind = torch.sum(atom_cpos * count_vec, dim=1)

        # atom neighbors' cell position and index
        atom_cnpos = atom_cpos.unsqueeze(1) + self.disp_mat
        atom_cnind = torch.sum(atom_cnpos * count_vec, dim=-1)
        
        # construct a C x N matrix to store the cell atom list, this is the most expensive part.
        padded_cind_sorted, padded_cind_args = torch.sort(padded_cind, stable=True)
        cell_ind, indices, cell_atom_num = torch.unique_consecutive(padded_cind_sorted, return_inverse=True, return_counts=True)
        max_cell_anum = int(cell_atom_num.max().item())
        global_cell_ind = torch.zeros(
            (num_cells, max_cell_anum, 2),
            dtype=c_pos_shap.dtype, 
            device=c_pos_shap.device,
        )
        cell_aind = torch.nonzero(torch.arange(max_cell_anum, device=count_vec.device).repeat(cell_atom_num.shape[0], 1) < cell_atom_num.unsqueeze(-1))[:, 1]
        global_cell_ind[padded_cind_sorted, cell_aind, 0] = padded_gind[padded_cind_args]
        global_cell_ind[padded_cind_sorted, cell_aind, 1] = padded_rind[padded_cind_args]

        # masking
        atom_nind = global_cell_ind[atom_cnind]
        pair_i, neigh, j = torch.where(atom_nind[:, :, :, 0])
        pair_j = atom_nind[pair_i, neigh, j, 1]
        pair_j_padded = atom_nind[pair_i, neigh, j, 0] - 1          # remember global index of padded atoms starts from 1
        pair_diff = padded_pos[pair_j_padded] - wrapped_pos[pair_i]

        pair_dist = torch.norm(pair_diff, dim = 1)
        mask = torch.logical_and(pair_dist < self.cutoff, pair_dist > 0.01)   # 0.01 for numerical stability
        pairs = torch.hstack((pair_i.unsqueeze(-1), pair_j.unsqueeze(-1)))
        
        # pairs包含原子对的索引，pair_diff是原子对之间的向量差，pair_dist是原子对之间的距离
        return pairs[mask].numpy(), pair_diff[mask].numpy(), pair_dist[mask].numpy()
    

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
    print("pair_diff形状：", pair_diff.shape)
    print("Pair differences:\n", pair_diff)
    print("Pair distances:\n", pair_dist)
    


def mdapy_neigh(atoms_obj: Atoms, cutoff:float):
    print("\n---------------mdapy_neigh:")
    import mdapy as mp
    from time import time
    mp.init('cpu')  # Use cpu, mp.init('gpu') will use gpu to compute.

    # start = time()
    # lattice_constant = 4.05
    # x, y, z = 3, 3, 3
    # FCC = mp.LatticeMaker(lattice_constant, "FCC", x, y, z)   # Create a FCC structure
    # FCC.compute()    # Get atom positions
    # end = time()
    # neigh = mp.Neighbor(FCC.pos, FCC.box, 5.0)     # Initialize Neighbor class.
    # neigh.compute()  # Calculate particle neighbor information.
    # print(neigh.verlet_list[0])               # Check the neighbor index.
    # print(neigh.neighbor_number[0])        # Check the neighbor distance.
    # print(neigh.distance_list[0])        # Check the neighbor atom number.
    

    
    # 检查晶胞大小是否满足条件
    cell_lengths = atoms_obj.cell.cellpar()[:3]
    print("cell_lengths = ", cell_lengths)
    assert all(cell_length >= 2 * cutoff for cell_length in cell_lengths), "晶胞尺寸应至少为截断半径的两倍"

    neigh = mp.Neighbor(atoms_obj.positions, atoms_obj.cell, cutoff)
    neigh.compute()
    print(neigh.verlet_list[0])       # Check the neighbor index.
    print(neigh.neighbor_number[0])   # Check the neighbor distance.
    print(neigh.distance_list[0])     # Check the neighbor atom number.
    
    # # 获得邻居列表
    # neighbors_list = _get_neighbors(atoms_obj, cutoff)

    # # 打印结果
    # for i, neighbors in enumerate(neighbors_list):
    #     print(f"Atom {i} has neighbors: {neighbors}")

    

def matscipy_neigh(atoms_obj: Atoms, cutoff: float):
    print("\n---------------matscipy:")
    from matscipy.neighbours import neighbour_list
    import test_examples1.neighborlist as neighborlist
    # pos = np.random.uniform(-4.0, 3.0, (100, 3))
    i1, j1, d1, D1 = neighborlist.neighbor_list_ijdD(positions=atoms_obj.positions, cutoff=cutoff, self_interaction=False)

    i = np.lexsort((j1, i1))
    # i1 = i1[i]
    # j1 = j1[i]
    # d1 = d1[i]
    # D1 = D1[i]
    # print("-------->ijdD")
    # print("i1[i] = \n", i1[i])
    # print("j1[i] = \n", j1[i])
    # print("d1[i] = \n", d1[i].shape)
    # print("D1[i] = \n", D1[i].shape)

    cutoff = cutoff
    # https://libatoms.github.io/matscipy/tools/neighbour_list.html
    atom_i, atom_j = neighbour_list("ij", positions=atoms_obj.positions, cutoff=cutoff)
    atomi, atomj, dist = neighbour_list("ijd", positions=atoms_obj.positions, cutoff=cutoff)
    i2, j2, d2, D2 = neighbour_list("ijdD", positions=atoms_obj.positions, cutoff=cutoff)

    i = np.lexsort((j2, i2))
    # i2 = i2[i]
    # j2 = j2[i]
    # d2 = d2[i]
    # D2 = D2[i]
    # print("-------->ij")
    # print("atom_i = \n", atom_i)
    # print("atom_j = \n", atom_j)
    print("-------->ijd")
    # print("atomi = \n", atomi)
    # print("atomj = \n", atomj)
    # pair_ij = [(i, j) for i, j in zip(atomi, atomj)]
    # print(pair_ij)
    # print('i = \n', i)
    print("i = \n", i2)
    print("j = \n", j2)
    print("dist = \n", dist)  # (i, j)的距离和(j, i)的距离 （各分一半）
    # print("-------->ijdD")
    # # print(i2, j2, d2, D2)
    # print("i = \n", i)
    # print("i2[i] = \n", i2[i])
    # print("j2[i] = \n", j2[i])
    # print("d2[i] = \n", d2[i].shape)
    # print("D2[i] = \n", D2[i].shape)



        

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

    # # 获得邻居列表
    # neighbors_list = _get_neighbors(atoms, cutoff)

    # # 打印结果
    # for i, neighbors in enumerate(neighbors_list):
    #     print(f"Atom {i} has neighbors: {neighbors}")


def torch_nl_neigh(atoms_obj: Atoms, cutoff: float):
    # https://gitlab.com/phd_learning/md_learning/torch_nl/-/blob/main/benchmark/benchmark.py?ref_type=heads
    print("\n---------------torch_nl_neigh------和ase计算结果一样:")
    from torch_nl import compute_neighborlist, ase2data
    frames = [atoms_obj]
    pos, cell, pbc, batch, n_atoms = ase2data(frames)
    mapping, batch_mapping, shifts_idx = compute_neighborlist(cutoff, pos, cell, pbc, batch, self_interaction=False)
    i = mapping[0,:] # indices of atoms
    j = mapping[1,:] # indices of neighbors
    D = atoms_obj.positions[j] - atoms_obj.positions[i] + np.array(shifts_idx).dot(atoms_obj.cell)  # via hjchen
    print("i = ", i)
    print("j = ", j)
    print("D.shape = ", D.shape)


def get_particle_pairs(atoms_obj: Atoms, cutoff: float):
    print("\n---------------ASE NeighborList:")

    # 创建邻居列表
    nl = NeighborList([cutoff / 2] * len(atoms_obj), self_interaction=False, bothways=True)

    # 更新邻居列表
    nl.update(atoms_obj)

    # 获取所有邻居对
    particle_pairs = []
    for i in range(len(atoms_obj)):
        indices, offsets = nl.get_neighbors(i)
        for index, offset in zip(indices, offsets):
            particle_pairs.append((i, index))

    # 输出结果
    print("--------> Particle Pairs")
    print("Particle pairs (i, j):")
    for pair in particle_pairs:
        print(pair)



# 定义一个函数来获取邻居列表------via hjchen
def _get_neighbors(atoms, cutoff):
    """
    获取每个原子的邻居列表。
    
    参数:
    atoms : ASE Atoms 对象
        包含所有原子位置的集合。
    cutoff : float
        邻居的最大距离。
        
    返回:
    neighbors : list of lists
        每个元素是一个列表，包含当前原子的邻居索引及其距离。
    """
    num_atoms = len(atoms)
    neighbors = [[] for _ in range(num_atoms)]
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair_diff = atoms.positions[i] - atoms.positions[j]
            distance = np.linalg.norm(atoms.positions[i] - atoms.positions[j])
            if distance <= cutoff:
                neighbors[i].append((j, pair_diff, distance))
                neighbors[j].append((i, pair_diff, distance))
                
    return neighbors


        

    
if __name__ == '__main__':
    
    """
        # 1. ase的近邻列表例子测试对比
        https://github.com/felixmusil/torch_nl/blob/main/benchmark/benchmark.py
        # 2. 使用ASE构建邻接矩阵：可用于图神经网络： 
        https://xnmao.github.io/blog/connectivity_matrix.html
        # 3. torch_nl 的例子测试
        https://gitlab.com/phd_learning/md_learning/torch_nl/-/blob/main/benchmark/benchmark.py?ref_type=heads
        # 机器学习势支持近邻列表计算的例子
        https://gitlab.com/phd_learning/md_learning/neighborlist/-/issues/2#note_1763588733
    """
    
    from ase import Atoms
    from ase.build import bulk, make_supercell
    from ase.neighborlist import NeighborList
    import numpy as np

    # 创建一个BCC晶胞的Atoms对象
    # 假设晶胞参数为a，原子类型为元素周期表号
    a = 3.32 # 晶胞参数，单位可以是埃
    element = 'Ta'  # 元素种类，例如Ta
    # 请创建一个bcc晶胞Atoms对象，用bulk命令
    atoms = bulk('Ta', 'bcc', a=a, cubic=True)

    # 定义扩展的比例
    # matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]  # 扩展比例矩阵
    matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # 扩展比例矩阵  SC222

    # 使用make_supercell方法扩展晶胞
    expanded_atoms = make_supercell(atoms, matrix)

    # 输出扩展后的晶胞信息
    print("扩展后的晶胞中原子数目:", len(expanded_atoms))
    print("扩展后的晶胞的单元矩阵:\n", expanded_atoms.get_cell())

    atoms_obj = expanded_atoms   # 赋值为一个超胞SC222
    # atoms_obj = atoms   # 赋值为一个单位晶胞SC111
    
    cutoff = 3.0   # 截断半径
    # cutoff = 6.150958970000000   # Ta的adp势文件中实际的截断半径
    TorchNeigh(atoms_obj, cutoff=cutoff)
    
    print()
    print("+++++++++++++++++++++++++++++++++++++++"*3)
    # 获得邻居列表
    neighbors_list = _get_neighbors(atoms_obj, cutoff)

    # 打印结果
    for i, neighbors in enumerate(neighbors_list):
        print(f"Atom {i} has neighbors: {neighbors}")
    print("+++++++++++++++++++++++++++++++++++++++"*3)

    
    # mdapy_neigh(atoms_obj, cutoff)
    # matscipy_neigh(atoms_obj, cutoff)
    ase_neigh(atoms_obj, cutoff)
    # get_particle_pairs(atoms_obj, cutoff)
    torch_nl_neigh(atoms_obj, cutoff)
    






