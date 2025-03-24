# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
pair_style      eam/fs

# 测试用（简）
#pair_coeff             * * ../../lmp_eam_fs_gen/Mg.eam.fs Mg

# 测试用（hjchen单独的fs.py生成代码测试）
#pair_coeff             * * ../../lmp_eam_fs_gen/Mg_2021test_chj.eam.fs  Mg


# 随机采样产生（真实情况）
pair_coeff             * * ../../lmp_eam_fs_gen/Mg_2021chj.eam.fs Mg
#pair_coeff             * * ./Mg_chj2021_select_idx(22)_10_1000_10qois.eam.fs Mg

# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no
