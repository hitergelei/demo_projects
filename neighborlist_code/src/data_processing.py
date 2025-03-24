import torch
import asap3 
from torch_neigh import TorchNeighborList
from ase import Atoms
from ase.io import read, write

#----------------------------------借鉴paiNN的代码模块功能------------------------------------
#   1. 封装atoms的数据类型，collate_fn函数，以及DataLoader
#   2. 自动微分求力
#   3. 能量和力的归一化处理
#   4. 保存和加载模型



# TODO: 得参考ase文档，实现自定义的ADP的calculator，然后才能获取atoms.get_potential_energy()和atoms.get_forces()的功能

class AseDataReader:
    def __init__(self, cutoff=5.0):            
        self.cutoff = cutoff
        # self.neighbor_list = asap3.FullNeighborList(self.cutoff, atoms=None)
        self.neighbor_list = TorchNeighborList(cutoff=cutoff)

    def __call__(self, atoms):
        atoms_data = {
            'num_atoms': torch.tensor([atoms.get_global_number_of_atoms()]),
            'elems': torch.tensor(atoms.numbers),
            'coord': torch.tensor(atoms.positions, dtype=torch.float),
        }
        atoms_data['image_idx'] = torch.zeros((atoms_data['num_atoms'],), dtype=torch.long)
        if atoms.pbc.any():            
            atoms_data['cell'] = torch.tensor(atoms.cell[:], dtype=torch.float)

        # pairs, n_diff = self.get_neighborlist(atoms)    
        # 计算近邻列表
        pairs, n_diff, pair_dist = self.neighbor_list(atoms)  # via hjchen
        atoms_data['pairs'] = torch.from_numpy(pairs)
        atoms_data['n_diff'] = torch.from_numpy(n_diff).float()
        atoms_data['num_pairs'] = torch.tensor([pairs.shape[0]])
        atoms_data['pair_dist'] = torch.from_numpy(pair_dist).float()   # via hjchen
        
        # 当atoms有能量和力输出时，读取这些输出。这些数据对训练模型是必要的，但是对预测不是必要的。
        try:
            energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)
            atoms_data['energy'] = energy
        except (AttributeError, RuntimeError):
            pass
        
        try: 
            forces = torch.tensor(atoms.get_forces(apply_constraint=False), dtype=torch.float)
            atoms_data['forces'] = forces
        except (AttributeError, RuntimeError):
            pass
        
        return atoms_data




