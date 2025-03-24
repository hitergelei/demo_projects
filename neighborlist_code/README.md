## LAMMPS2024版本python以及C++的API接口安装
### 复制一份虚拟环境

```sh
(GNN_painn) [centos@localhost GNN_painn]$ conda create -n neighlist_env --clone GNN_painn
Source:      /home/centos/anaconda3/envs/GNN_painn
Destination: /home/centos/anaconda3/envs/neighlist_env
Packages: 94
Files: 6141
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate neighlist_env
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### 启动虚拟环境

```sh
(GNN_painn) [centos@localhost GNN_painn]$ conda activate neighlist_env
(neighlist_env) [centos@localhost GNN_painn]$ 
(neighlist_env) [centos@localhost GNN_painn]$ 
```

### 安装依赖包

#### matscipy库

`matscipy-1.1.0`依赖于 `ase-3.23.0`而不是 `ase-3.22.1`

```sh
(neighlist_env) [centos@localhost GNN_painn]$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matscipy
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting matscipy
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/79/d9/0c42c9cd653c021cacd43756abb68e0a22183eeb46cb9da8e960f5aeaa3c/matscipy-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (438 kB)
Requirement already satisfied: numpy>=1.16.0 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matscipy) (1.26.4)
Requirement already satisfied: scipy>=1.2.3 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matscipy) (1.14.0)
Collecting ase>=3.23.0 (from matscipy)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/02/81/2c339c920fb1be1caa0b7efccb14452c9f4f0dbe3837f33519610611f57b/ase-3.23.0-py3-none-any.whl (2.9 MB)
Requirement already satisfied: packaging in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matscipy) (24.1)
Requirement already satisfied: matplotlib>=3.3.4 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from ase>=3.23.0->matscipy) (3.9.1)
Requirement already satisfied: contourpy>=1.0.1 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matplotlib>=3.3.4->ase>=3.23.0->matscipy) (1.2.1)
Requirement already satisfied: cycler>=0.10 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matplotlib>=3.3.4->ase>=3.23.0->matscipy) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matplotlib>=3.3.4->ase>=3.23.0->matscipy) (4.53.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matplotlib>=3.3.4->ase>=3.23.0->matscipy) (1.4.5)
Requirement already satisfied: pillow>=8 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matplotlib>=3.3.4->ase>=3.23.0->matscipy) (9.0.0)
Requirement already satisfied: pyparsing>=2.3.1 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matplotlib>=3.3.4->ase>=3.23.0->matscipy) (3.1.2)
Requirement already satisfied: python-dateutil>=2.7 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from matplotlib>=3.3.4->ase>=3.23.0->matscipy) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib>=3.3.4->ase>=3.23.0->matscipy) (1.16.0)
Installing collected packages: ase, matscipy
  Attempting uninstall: ase
    Found existing installation: ase 3.22.1
    Uninstalling ase-3.22.1:
      Successfully uninstalled ase-3.22.1
Successfully installed ase-3.23.0 matscipy-1.1.0
(neighlist_env) [centos@localhost GNN_painn]$ 
```

#### mdapy库

```sh
(neighlist_env) [centos@localhost GNN_painn]$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mdapy
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting mdapy
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f1/64/577785c36792ffa802d046231672892822f76e2d427cec378ac6c09a93d1/mdapy-0.11.3.tar.gz (585 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 585.6/585.6 kB 4.3 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  ...
  ...
    Stored in directory: /home/centos/.cache/pip/wheels/c7/8e/55/ba097d346d73b51bb2a76827a1338031aab270c46552cd0539
Successfully built mdapy
Installing collected packages: tqdm, polyscope, polars, mdurl, dill, colorama, markdown-it-py, rich, taichi, mdapy
Successfully installed colorama-0.4.6 dill-0.3.8 markdown-it-py-3.0.0 mdapy-0.11.3 mdurl-0.1.2 polars-1.8.2 polyscope-2.3.0 rich-13.8.1 taichi-1.7.2 tqdm-4.66.5
```

#### torch-nl库

```sh
git clone https://github.com/felixmusil/torch_nl.git
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple black[jupyter]
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest
pip install -r requirements.txt 
```

### 运行

```txt
在代码中，pairs数组包含了原子对的索引。每个元素是一个长度为2的数组，表示一对原子的索引。例如，[0, 1]表示原子0和原子1是一对邻居，[1, 0]表示原子1和原子0是一对邻居。

具体来说：

[0, 1]表示原子0的一个邻居是原子1。
[1, 0]表示原子1的一个邻居是原子0。
在邻居列表中，通常会记录每个原子的所有邻居。由于邻居关系是对称的（如果原子A是原子B的邻居，那么原子B也是原子A的邻居），所以会出现[0, 1]和[1, 0]这样的对称对。

在你的代码中，pairs数组中包含了多个[0, 1]和[1, 0]，这可能是因为在计算邻居列表时，考虑了不同的周期性边界条件或不同的图像原子。每个[0, 1]和[1, 0]对可能对应于不同的周期性边界条件下的邻居关系。

如果你希望减少重复的邻居对，可以在生成邻居列表时进行去重处理。
```

---

### LAMMPS2024版本python的API接口安装

> 由于我们需要实现ADP势函数的解析形式，用于自动微分求导。最方便的使用lammps提供的python版api接口（2024版增加了很多新功能），然后借助pytorch实现的可微分求导的近邻列表代码，使得在拟合EFS目标量时，误差反向传播进而优化ADP势函数的参数

#### 1. 安装步骤（基于GPU节点工作站）

> 参考我自己写的飞书教程：[python的lammps接口安装和测试](https://hi0f2f0g55.feishu.cn/wiki/LnZ1wCLTeiWl59kkIePc9xXintX?from=from_copylink)

##### 1.1 准备好lammps版本和对应的并行编译器

> 注：如果没有英特尔并行编译器或者mpich并行库，可以考虑用conda 安装：`conda install -c conda-forge mpich`（这个没测过，理论可行的）

```sh
(neighlist_env) [centos@localhost software]$ tar -xvf lammps_29Aug2024.tar.gz 
(neighlist_env) [centos@localhost src]$ pwd
/home/centos/hjchen/software/lammps-29Aug2024/src
# --------我事先安装过支持mpicxx的库mpich-4.1，可以不用考虑用英特尔的并行编译器了
(neighlist_env) [centos@localhost src]$ which mpicxx    
~/hjchen/mpich-4.1/_build/bin/mpicxx
```

##### 1.2 编译lammps的动态链接库

```sh
(neighlist_env) [centos@localhost src]$ make yes-MANYBODY
Installing package MANYBODY
(neighlist_env) [centos@localhost src]$ make mode=shared mpi
Gathering installed package information (may take a little while)
make[1]: 进入目录“/home/centos/hjchen/software/lammps-29Aug2024/src”
Gathering git version information
make[1]: 离开目录“/home/centos/hjchen/software/lammps-29Aug2024/src”
Compiling LAMMPS for machine mpi
make[1]: 进入目录“/home/centos/hjchen/software/lammps-29Aug2024/src/Obj_shared_mpi”
cc -O -o fastdep.exe ../DEPEND/fastdep.c
make[1]: 离开目录“/home/centos/hjchen/software/lammps-29Aug2024/src/Obj_shared_mpi”
make[1]: 进入目录“/home/centos/hjchen/software/lammps-29Aug2024/src/Obj_shared_mpi”
mpicxx -g -O3 -std=c++11 -fPIC -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64    -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX=1     -c ../main.cpp
mpicxx -g -O3 -std=c++11 -fPIC -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64    -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX=1     -c ../write_restart.cpp
mpicxx -g -O3 -std=c++11 -fPIC -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64    -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX=1     -c ../compute_temp_deform.cpp
mpicxx -g -O3 -std=c++11 -fPIC -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64    -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX=1     -c ../write_coeff.cpp
mpicxx -g -O3 -std=c++11 -fPIC -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64    -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX=1     -c ../velocity.cpp
mpicxx -g -O3 -std=c++11 -fPIC -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64    -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX=1     -c ../min_cg.cpp
。。。
。。。
dummy.o balance.o compute_pe_atom.o npair_skip_size_off2on.o min_hftn.o compute_temp_sphere.o pair_zero.o compute_heat_flux.o compute_temp_region.o angle_zero.o npair_multi.o pair_rebomos.o npair_nsq.o fix_wall_harmonic.o atom_vec.o pair_tersoff_zbl.o bond_hybrid.o compute_stress_atom.o fix_respa.o integrate.o nstencil_ghost_bin.o pair_comb.o pair_eam.o pair_sw_mod.o dump_local.o       -ldl 
mpicxx -g -O3 -std=c++11 main.o       -L. -llammps_mpi       -ldl  -o ../lmp_mpi
size ../lmp_mpi
   text    data     bss     dec     hex filename
   4893     936       8    5837    16cd ../lmp_mpi
make[1]: 离开目录“/home/centos/hjchen/software/lammps-29Aug2024/src/Obj_shared_mpi”
```

可以看到，此时src目录下有liblammps.so 和 liblammps_mpi.so 两个动态链接库。其中，liblammps.so是liblammps_mpi.so的软连接（快捷键）

```sh
(neighlist_env) [centos@localhost src]$ ll liblammps*
-rwxrwxr-x. 1 centos centos 124381152 9月  26 22:34 liblammps_mpi.so
lrwxrwxrwx. 1 centos centos        16 9月  26 22:34 liblammps.so -> liblammps_mpi.so
(neighlist_env) [centos@localhost src]$ 
```

##### 1.3 编译lammps的python接口

> 由于我们使用的虚拟环境里面调用lammps的python接口，需要执行如下安装命令：
> `make install-python`  （该命令需要VPN，才能获得对pypi的请求，如果没有，则需要自己设置pip的镜像源为清华镜像）

```sh
#----为了方便下载，设置pip install的镜像源为清华镜像源
(neighlist_env) [centos@localhost src]$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
Writing to /home/centos/.config/pip/pip.conf
(neighlist_env) [centos@localhost src]$ 
(neighlist_env) [centos@localhost src]$ make install-python
Purging existing wheels...

Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Requirement already satisfied: pip in ./buildwheel/lib/python3.10/site-packages (23.0.1)
Collecting pip
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d4/55/90db48d85f7689ec6f81c0db0622d704306c5284850383c090e6c7195a5c/pip-24.2-py3-none-any.whl (1.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 4.2 MB/s eta 0:00:00
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 23.0.1
    Uninstalling pip-23.0.1:
      Successfully uninstalled pip-23.0.1
Successfully installed pip-24.2
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Requirement already satisfied: pip in ./buildwheel/lib/python3.10/site-packages (from -r wheel_requirements.txt (line 1)) (24.2)
Collecting build (from -r wheel_requirements.txt (line 2))
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/91/fd/e4bda6228637ecae5732162b5ac2a5a822e2ba8e546eb4997cde51b231a3/build-1.2.2-py3-none-any.whl (22 kB)
Collecting wheel (from -r wheel_requirements.txt (line 3))
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/1b/d1/9babe2ccaecff775992753d8686970b1e2755d21c8a63be73aba7a4e7d77/wheel-0.44.0-py3-none-any.whl (67 kB)
Requirement already satisfied: setuptools in ./buildwheel/lib/python3.10/site-packages (from -r wheel_requirements.txt (line 4)) (65.5.0)
Collecting setuptools (from -r wheel_requirements.txt (line 4))
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/ff/ae/f19306b5a221f6a436d8f2238d5b80925004093fa3edea59835b514d9057/setuptools-75.1.0-py3-none-any.whl (1.2 MB)
Collecting packaging>=19.1 (from build->-r wheel_requirements.txt (line 2))
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/08/aa/cc0199a5f0ad350994d660967a8efb233fe0416e4639146c089643407ce6/packaging-24.1-py3-none-any.whl (53 kB)
Collecting pyproject_hooks (from build->-r wheel_requirements.txt (line 2))
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ae/f3/431b9d5fe7d14af7a32340792ef43b8a714e7726f1d7b69cc4e8e7a3f1d7/pyproject_hooks-1.1.0-py3-none-any.whl (9.2 kB)
Collecting tomli>=1.1.0 (from build->-r wheel_requirements.txt (line 2))
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/97/75/10a9ebee3fd790d20926a90a2547f0bf78f371b2f13aa822c759680ca7b9/tomli-2.0.1-py3-none-any.whl (12 kB)
Installing collected packages: wheel, tomli, setuptools, pyproject_hooks, packaging, build
  Attempting uninstall: setuptools
    Found existing installation: setuptools 65.5.0
    Uninstalling setuptools-65.5.0:
      Successfully uninstalled setuptools-65.5.0
Successfully installed build-1.2.2 packaging-24.1 pyproject_hooks-1.1.0 setuptools-75.1.0 tomli-2.0.1 wheel-0.44.0
Building new binary wheel
* Getting build dependencies for wheel...
running egg_info
creating lammps.egg-info
writing lammps.egg-info/PKG-INFO
writing dependency_links to lammps.egg-info/dependency_links.txt
writing top-level names to lammps.egg-info/top_level.txt
writing manifest file 'lammps.egg-info/SOURCES.txt'
reading manifest file 'lammps.egg-info/SOURCES.txt'
writing manifest file 'lammps.egg-info/SOURCES.txt'
* Building wheel...
running bdist_wheel
running build
running build_py
creating build/lib.linux-x86_64-cpython-310/lammps
copying lammps/__init__.py -> build/lib.linux-x86_64-cpython-310/lammps
copying lammps/constants.py -> build/lib.linux-x86_64-cpython-310/lammps
copying lammps/core.py -> build/lib.linux-x86_64-cpython-310/lammps
copying lammps/data.py -> build/lib.linux-x86_64-cpython-310/lammps
copying lammps/formats.py -> build/lib.linux-x86_64-cpython-310/lammps
copying lammps/numpy_wrapper.py -> build/lib.linux-x86_64-cpython-310/lammps
copying lammps/pylammps.py -> build/lib.linux-x86_64-cpython-310/lammps
creating build/lib.linux-x86_64-cpython-310/lammps/mliap
copying lammps/mliap/__init__.py -> build/lib.linux-x86_64-cpython-310/lammps/mliap
copying lammps/mliap/loader.py -> build/lib.linux-x86_64-cpython-310/lammps/mliap
copying lammps/mliap/mliap_unified_abc.py -> build/lib.linux-x86_64-cpython-310/lammps/mliap
copying lammps/mliap/mliap_unified_lj.py -> build/lib.linux-x86_64-cpython-310/lammps/mliap
copying lammps/mliap/pytorch.py -> build/lib.linux-x86_64-cpython-310/lammps/mliap
copying lammps/liblammps.so -> build/lib.linux-x86_64-cpython-310/lammps
running build_ext
installing to build/bdist.linux-x86_64/wheel
running install
running install_lib
creating build/bdist.linux-x86_64/wheel
creating build/bdist.linux-x86_64/wheel/lammps
copying build/lib.linux-x86_64-cpython-310/lammps/__init__.py -> build/bdist.linux-x86_64/wheel/./lammps
copying build/lib.linux-x86_64-cpython-310/lammps/constants.py -> build/bdist.linux-x86_64/wheel/./lammps
copying build/lib.linux-x86_64-cpython-310/lammps/core.py -> build/bdist.linux-x86_64/wheel/./lammps
copying build/lib.linux-x86_64-cpython-310/lammps/data.py -> build/bdist.linux-x86_64/wheel/./lammps
copying build/lib.linux-x86_64-cpython-310/lammps/formats.py -> build/bdist.linux-x86_64/wheel/./lammps
copying build/lib.linux-x86_64-cpython-310/lammps/numpy_wrapper.py -> build/bdist.linux-x86_64/wheel/./lammps
copying build/lib.linux-x86_64-cpython-310/lammps/pylammps.py -> build/bdist.linux-x86_64/wheel/./lammps
creating build/bdist.linux-x86_64/wheel/lammps/mliap
copying build/lib.linux-x86_64-cpython-310/lammps/mliap/__init__.py -> build/bdist.linux-x86_64/wheel/./lammps/mliap
copying build/lib.linux-x86_64-cpython-310/lammps/mliap/loader.py -> build/bdist.linux-x86_64/wheel/./lammps/mliap
copying build/lib.linux-x86_64-cpython-310/lammps/mliap/mliap_unified_abc.py -> build/bdist.linux-x86_64/wheel/./lammps/mliap
copying build/lib.linux-x86_64-cpython-310/lammps/mliap/mliap_unified_lj.py -> build/bdist.linux-x86_64/wheel/./lammps/mliap
copying build/lib.linux-x86_64-cpython-310/lammps/mliap/pytorch.py -> build/bdist.linux-x86_64/wheel/./lammps/mliap
copying build/lib.linux-x86_64-cpython-310/lammps/liblammps.so -> build/bdist.linux-x86_64/wheel/./lammps
running install_egg_info
running egg_info
writing lammps.egg-info/PKG-INFO
writing dependency_links to lammps.egg-info/dependency_links.txt
writing top-level names to lammps.egg-info/top_level.txt
reading manifest file 'lammps.egg-info/SOURCES.txt'
writing manifest file 'lammps.egg-info/SOURCES.txt'
Copying lammps.egg-info to build/bdist.linux-x86_64/wheel/./lammps-2024.8.29-py3.10.egg-info
running install_scripts
creating build/bdist.linux-x86_64/wheel/lammps-2024.8.29.dist-info/WHEEL
creating '/home/centos/hjchen/software/lammps-29Aug2024/src/build-python/.tmp-hbku868d/lammps-2024.8.29-cp310-cp310-linux_x86_64.whl' and adding 'build/bdist.linux-x86_64/wheel' to it
adding 'lammps/__init__.py'
adding 'lammps/constants.py'
adding 'lammps/core.py'
adding 'lammps/data.py'
adding 'lammps/formats.py'
adding 'lammps/liblammps.so'
adding 'lammps/numpy_wrapper.py'
adding 'lammps/pylammps.py'
adding 'lammps/mliap/__init__.py'
adding 'lammps/mliap/loader.py'
adding 'lammps/mliap/mliap_unified_abc.py'
adding 'lammps/mliap/mliap_unified_lj.py'
adding 'lammps/mliap/pytorch.py'
adding 'lammps-2024.8.29.dist-info/METADATA'
adding 'lammps-2024.8.29.dist-info/WHEEL'
adding 'lammps-2024.8.29.dist-info/top_level.txt'
adding 'lammps-2024.8.29.dist-info/RECORD'
removing build/bdist.linux-x86_64/wheel
Successfully built lammps-2024.8.29-cp310-cp310-linux_x86_64.whl
Installing wheel into system site-packages folder
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Processing ./lammps-2024.8.29-cp310-cp310-linux_x86_64.whl
Installing collected packages: lammps
Successfully installed lammps-2024.8.29
(neighlist_env) [centos@localhost src]$ 
```

我们可以查看下neighlist_env虚拟环境下的lammps的动态链接库liblammps.so：

```sh
(neighlist_env) [centos@localhost src]$ ll ~/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps
总用量 121640
-rw-rw-r--. 1 centos centos      1779 9月  26 22:47 constants.py
-rw-rw-r--. 1 centos centos     94354 9月  26 22:47 core.py
-rw-rw-r--. 1 centos centos      3163 9月  26 22:47 data.py
-rw-rw-r--. 1 centos centos      8007 9月  26 22:47 formats.py
-rw-rw-r--. 1 centos centos      1498 9月  26 22:47 __init__.py
-rwxrwxr-x. 1 centos centos 124381152 9月  26 22:47 liblammps.so
drwxrwxr-x. 3 centos centos       134 9月  26 22:47 mliap
-rw-rw-r--. 1 centos centos     20751 9月  26 22:47 numpy_wrapper.py
drwxrwxr-x. 2 centos centos       227 9月  26 22:47 __pycache__
-rw-rw-r--. 1 centos centos     31353 9月  26 22:47 pylammps.py
(neighlist_env) [centos@localhost src]$ 
(neighlist_env) [centos@localhost src]$ 
```

通过diff命令可以看到，这两个动态链接库是一样的

```sh
(neighlist_env) [centos@localhost src]$ 
(neighlist_env) [centos@localhost src]$ diff  ~/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps/liblammps.so   liblammps.so 
(neighlist_env) [centos@localhost src]$ 

#--------我们还可以ldd查看下liblammps.so动态链接库的依赖------------
(neighlist_env) [centos@localhost src]$ ldd liblammps.so 
        linux-vdso.so.1 (0x00007ffccd951000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007fce31c01000)
        libmpicxx.so.12 => /home/centos/hjchen/mpich-4.1/_build/lib/libmpicxx.so.12 (0x00007fce319df000)
        libmpi.so.12 => /home/centos/hjchen/mpich-4.1/_build/lib/libmpi.so.12 (0x00007fce2eeae000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007fce2eadb000)
        libm.so.6 => /lib64/libm.so.6 (0x00007fce2e759000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007fce2e541000)
        libc.so.6 => /lib64/libc.so.6 (0x00007fce2e17c000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fce32888000)
        libatomic.so.1 => /usr/local/lib/../lib64/libatomic.so.1 (0x00007fce2df74000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007fce2dd54000)
        librt.so.1 => /lib64/librt.so.1 (0x00007fce2db4c000)
(neighlist_env) [centos@localhost src]$ 
```

测试下，可以看到，lammps的python接口已经成功了：

```py
(neighlist_env) [centos@localhost src]$ ipython
Python 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.25.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import lammps

In [2]: lmp = lammps.lammps()
LAMMPS (29 Aug 2024)

In [3]: 
```

##### 1.4 mpi4py并行地运行LAMMPS

> 得下载mpi4py库，如果pip命令安装有问题，考虑用conda命令安装，环境依赖更加独立：`conda install mpi4py`

```sh
(neighlist_env) [centos@localhost src]$ conda  install mpi4py
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.12.0
  latest version: 24.7.1

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /home/centos/anaconda3/envs/neighlist_env

  added / updated specs:
    - mpi4py


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    certifi-2024.8.30          |  py310h06a4308_0         162 KB
    mpi4py-3.1.4               |  py310hfc96bbd_0         573 KB
    ------------------------------------------------------------
                                           Total:         735 KB

The following NEW packages will be INSTALLED:

  libgfortran-ng     pkgs/main/linux-64::libgfortran-ng-7.5.0-ha8ba4b0_17
  libgfortran4       pkgs/main/linux-64::libgfortran4-7.5.0-ha8ba4b0_17
  mpi                pkgs/main/linux-64::mpi-1.0-mpich
  mpi4py             pkgs/main/linux-64::mpi4py-3.1.4-py310hfc96bbd_0
  mpich              pkgs/main/linux-64::mpich-3.3.2-hc856adb_0

The following packages will be UPDATED:

  certifi                          2024.7.4-py310h06a4308_0 --> 2024.8.30-py310h06a4308_0
  openssl                                 3.0.14-h5eee18b_0 --> 3.0.15-h5eee18b_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
mpi4py-3.1.4         | 573 KB    | ############################################################################################## | 100% 
certifi-2024.8.30    | 162 KB    | ############################################################################################## | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
(neighlist_env) [centos@localhost src]$ pip list
Package           Version
----------------- -----------
ase               3.23.0
asttokens         2.0.5
black             24.8.0
certifi           2024.8.30
click             8.1.7
colorama          0.4.6
comm              0.2.1
contourpy         1.2.1
cycler            0.12.1
debugpy           1.6.7
decorator         5.1.1
dill              0.3.8
exceptiongroup    1.2.0
executing         0.8.3
filelock          3.13.1
fonttools         4.53.1
gmpy2             2.1.2
iniconfig         2.0.0
ipykernel         6.29.5
ipython           8.25.0
jedi              0.19.1
Jinja2            3.1.4
jupyter_client    8.6.0
jupyter_core      5.7.2
kiwisolver        1.4.5
lammps            2024.8.29
markdown-it-py    3.0.0
MarkupSafe        2.1.3
matplotlib        3.9.1
matplotlib-inline 0.1.6
matscipy          1.1.0
mdapy             0.11.3
mdurl             0.1.2
mkl-fft           1.3.8
mkl-random        1.2.4
mkl-service       2.4.0
mpi4py            3.1.4
mpmath            1.3.0
mypy-extensions   1.0.0
nest-asyncio      1.6.0
networkx          3.3
numpy             1.26.4
packaging         24.1
parso             0.8.3
pathspec          0.12.1
pexpect           4.8.0
Pillow            9.0.0
pip               24.0
platformdirs      3.10.0
pluggy            1.5.0
polars            1.8.2
polyscope         2.3.0
prompt-toolkit    3.0.43
psutil            5.9.0
ptyprocess        0.7.0
pure-eval         0.2.2
Pygments          2.15.1
pyparsing         3.1.2
pytest            8.3.3
python-dateutil   2.9.0.post0
pyzmq             25.1.2
rich              13.8.1
scipy             1.14.0
setuptools        69.5.1
six               1.16.0
stack-data        0.2.0
sympy             1.12
taichi            1.7.2
tokenize-rt       6.0.0
tomli             2.0.1
torch             2.0.1
torch_nl          0.3
tornado           6.4.1
tqdm              4.66.5
traitlets         5.14.3
triton            2.0.0
typing_extensions 4.11.0
wcwidth           0.2.5
wheel             0.43.0
(neighlist_env) [centos@localhost src]$ which mpirun
~/anaconda3/envs/neighlist_env/bin/mpirun
(neighlist_env) [centos@localhost src]$ 
```

测试下代码 test_lammps_python_mpi_api.py：

> 注：如果想并行执行python文件: 则使用命令 `mpirun -n 4 python test_lammps_python_mpi_api.py`

```py
import lammps
from mpi4py import MPI
import subprocess

from lammps import lammps

# NOTE: argv[0] is set by the lammps class constructor
args = ["-log", "none"]

# create LAMMPS instance
lmp = lammps(cmdargs=args)

# get and print numerical version code
print("LAMMPS Version: ", lmp.version())

# explicitly close and delete LAMMPS instance (optional)
lmp.close()

comm = MPI.COMM_WORLD
print("Proc %d out of %d procs" % (comm.Get_rank(),comm.Get_size()))

print("------------------------------------------------------------------")
import torch

# 检查 CPU 和 GPU 信息
cpu_info = torch.__version__
print(f"PyTorch Version: {torch.__version__}")
print(f"CPU Information: {torch.version.cuda}")

# 检查 GPU 信息
if torch.cuda.is_available():
    print("CUDA is available.")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
    print(f"CUDA Current Device: {torch.cuda.current_device()}")

    # 获取 NVIDIA CUDA 驱动版本
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True)
        driver_version = result.stdout.strip()
        print(f"NVIDIA CUDA Driver Version: {driver_version}")
    except Exception as e:
        print(f"Failed to get NVIDIA CUDA Driver Version: {e}")
else:
    print("CUDA is not available.")
    print("NVIDIA GPU and CUDA driver information is not available.")
```

运行结果：

```py
LAMMPS (29 Aug 2024)
LAMMPS Version:  20240829
Total wall time: 0:00:00
Proc 0 out of 1 procs
------------------------------------------------------------------
PyTorch Version: 2.0.1
CPU Information: 11.7
CUDA is available.
CUDA Version: 11.7
CUDA Device Name: NVIDIA GeForce RTX 3080 Ti
Number of CUDA Devices: 1
CUDA Current Device: 0
NVIDIA CUDA Driver Version: 515.76
```

并行执行python文件: `mpirun -n 4 python test_lammps_python_mpi_api.py`

```sh
(neighlist_env) [centos@localhost neighborlist_code]$ which mpirun
~/anaconda3/envs/neighlist_env/bin/mpirun
(neighlist_env) [centos@localhost neighborlist_code]$ mpirun -n 4 python test_lammps_python_mpi_api.py 
LAMMPS (29 Aug 2024)
WARNING: Using I/O redirection is unreliable with parallel runs. Better to use the -in switch to read input files. (../lammps.cpp:571)
Total wall time: 0:00:00
Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
LAMMPS Version:  20240829
Proc 0 out of 4 procs
------------------------------------------------------------------
PyTorch Version: 2.0.1
CPU Information: 11.7
CUDA is available.
CUDA Version: 11.7
CUDA Device Name: NVIDIA GeForce RTX 3080 Ti
Number of CUDA Devices: 1
CUDA Current Device: 0
NVIDIA CUDA Driver Version: 515.76
Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
LAMMPS Version:  20240829
Proc 1 out of 4 procs
------------------------------------------------------------------
PyTorch Version: 2.0.1
CPU Information: 11.7
CUDA is available.
CUDA Version: 11.7
CUDA Device Name: NVIDIA GeForce RTX 3080 Ti
Number of CUDA Devices: 1
CUDA Current Device: 0
NVIDIA CUDA Driver Version: 515.76
Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
LAMMPS Version:  20240829
Proc 3 out of 4 procs
------------------------------------------------------------------
PyTorch Version: 2.0.1
CPU Information: 11.7
CUDA is available.
CUDA Version: 11.7
CUDA Device Name: NVIDIA GeForce RTX 3080 Ti
Number of CUDA Devices: 1
CUDA Current Device: 0
NVIDIA CUDA Driver Version: 515.76
Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
LAMMPS Version:  20240829
Proc 2 out of 4 procs
------------------------------------------------------------------
PyTorch Version: 2.0.1
CPU Information: 11.7
CUDA is available.
CUDA Version: 11.7
CUDA Device Name: NVIDIA GeForce RTX 3080 Ti
Number of CUDA Devices: 1
CUDA Current Device: 0
NVIDIA CUDA Driver Version: 515.76
(neighlist_env) [centos@localhost neighborlist_code]$ 
```

#### 2. 在jupyter notebook中调用lammps

##### 2.1 为jupyter notebook添加内核

> Q: 如果在终端输入jupyter notebook时，浏览器跳转后，发现没有我们所需要的虚拟环境的kernel，怎么办？  
> A: [Anaconda配置虚拟环境并为Jupyter notebook添加内核｜功能](https://blog.csdn.net/weixin_41429931/article/details/113623640)

```sh
(neighlist_env) [centos@localhost GNN_painn]$ 
(neighlist_env) [centos@localhost GNN_painn]$ conda install -c anaconda ipykernel
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.12.0
  latest version: 24.7.1

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /home/centos/anaconda3/envs/neighlist_env

  added / updated specs:
    - ipykernel


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ca-certificates-2024.7.2   |       h06a4308_0         133 KB  anaconda
    certifi-2024.8.30          |  py310h06a4308_0         164 KB  anaconda
    ipykernel-6.28.0           |  py310h06a4308_0         181 KB  anaconda
    openssl-3.0.15             |       h5eee18b_0         5.2 MB  anaconda
    ------------------------------------------------------------
                                           Total:         5.7 MB

The following packages will be SUPERSEDED by a higher-priority channel:

  ca-certificates                                 pkgs/main --> anaconda
  certifi                                         pkgs/main --> anaconda
  ipykernel          conda-forge/noarch::ipykernel-6.29.5-~ --> anaconda/linux-64::ipykernel-6.28.0-py310h06a4308_0
  openssl                                         pkgs/main --> anaconda


Proceed ([y]/n)? y


Downloading and Extracting Packages
ca-certificates-2024 | 133 KB    | #################################################################################################################### | 100% 
certifi-2024.8.30    | 164 KB    | #################################################################################################################### | 100% 
ipykernel-6.28.0     | 181 KB    | #################################################################################################################### | 100% 
openssl-3.0.15       | 5.2 MB    | #################################################################################################################### | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
(neighlist_env) [centos@localhost GNN_painn]$
```

将虚拟环境的kernel（这里名字就叫 `neighlist_env`），添加到jupyter notebook中

```sh
(neighlist_env) [centos@localhost GNN_painn]$ python -m ipykernel install --user --name=neighlist_env
Installed kernelspec neighlist_env in /home/centos/.local/share/jupyter/kernels/neighlist_env
(neighlist_env) [centos@localhost GNN_painn]$ 
```

还可以为 jupyter notebook添加目录功能: `pip install jupyter_contrib_nbextensions`

---

### LAMMPS2024版本C++的API接口编译

由于已经有了lammps的动态链接库 `liblammps.so`，只需要把我们自己写的C++代码和这个动态链接库一起编译就可以了。

`lammps_cplusplus_API_ex.cpp`代码如下：

```cpp
#include "lammps.h"

#include <mpi.h>
#include <iostream>


// 怎么编译？https://docs.lammps.org/Build_link.html

int main(int argc, char **argv)
{
    LAMMPS_NS::LAMMPS *lmp;
    // custom argument vector for LAMMPS library
    const char *lmpargv[] = { "liblammps", "-log", "none"};  
    int lmpargc = sizeof(lmpargv)/sizeof(const char *);

    // explicitly initialize MPI
    MPI_Init(&argc, &argv);

    // create LAMMPS instance
    lmp = new LAMMPS_NS::LAMMPS(lmpargc, (char **)lmpargv, MPI_COMM_WORLD);
    // output numerical version string
    std::cout << "LAMMPS version ID: " << lmp->num_ver << std::endl;
    // delete LAMMPS instance
    delete lmp;

    // stop MPI environment
    MPI_Finalize();
    return 0;
}
```

**step(0): 启动虚拟环境 `neighlist_env`**

```sh
(base) [centos@localhost GNN_painn]$ conda deactivate
[centos@localhost GNN_painn]$ conda activate neighlist_env
(neighlist_env) [centos@localhost GNN_painn]$ python -V
Python 3.10.14
(neighlist_env) [centos@localhost GNN_painn]$ 
```

**step(1): 第一个关键的地方，需要引入lammps的动态库（`liblammps.so`）的环境变量，通过 `export LD_LIBRARY_PATH`命令**：
你可以这个写

```sh
(neighlist_env) [centos@localhost neighborlist_code]$ export LD_LIBRARY_PATH=/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps:$LD_LIBRARY_PATH
```

或者这样写

```sh
(neighlist_env) [centos@localhost neighborlist_code]$ export LD_LIBRARY_PATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$LD_LIBRARY_PATH
```

**step(2): 第二个最关键的地方，要引入mpicxx的环境（不要使用conda虚拟环境中的mpicxx（好像是有问题的），而是必须使用mpich中的mpicxx）**

```sh
(neighlist_env) [centos@localhost GNN_painn]$ which mpicxx
~/hjchen/mpich-4.1/_build/bin/mpicxx
(neighlist_env) [centos@localhost GNN_painn]$ 
```

接着，在include( `-I` 操作)这个lammps.h头文件和和link( `-L` 操作)这个`liblammps.so`链接库路径时，你可以这样写

```sh
(neighlist_env) [centos@localhost neighborlist_code]$ /home/centos/hjchen/mpich-4.1/_build/bin/mpicxx lammps_cpluplus_API_ex.cpp -o lmp_cplusplus_api_ex -I/home/centos/hjchen/software/lammps-29Aug2024/src -L/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps -llammps
(neighlist_env) [centos@localhost neighborlist_code]$ ./lmp_cplusplus_api_ex 
LAMMPS (29 Aug 2024)
LAMMPS version ID: 20240829
Total wall time: 0:00:00
```

或者这样写

```sh
(neighlist_env) [centos@localhost neighborlist_code]$ /home/centos/hjchen/mpich-4.1/_build/bin/mpicxx lammps_cpluplus_API_ex.cpp -o lmp_cplusplus_api_ex -I/home/centos/hjchen/software/lammps-29Aug2024/src/ -L/home/centos/hjchen/software/lammps-29Aug2024/src/ -llammps
```

---

**注：如果你想把上面step(0)~step(2)步骤都省了，可以直接把mpicxx和lammps的环境都加进去**
>参考：[详解Linux下环境变量C_INCLUDE_PATH、CPLUS_INCLUDE_PATH、CPATH以及常见错误](https://blog.csdn.net/weixin_44327262/article/details/105860213)  

```sh
conda deactivate
conda activate neighlist_env
export PATH=/home/centos/hjchen/mpich-4.1/_build/bin:$PATH  
export CPATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$CPATH  
export LD_LIBRARY_PATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$LD_LIBRARY_PATH   # 用于运行时的环境
# 或者下面的也行
#export LD_LIBRARY_PATH=/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps:$LD_LIBRARY_PATH

# --->然后编译时执行命令
mpicxx lammps_cplusplus_API_ex.cpp   -o lmp_cpp_api -L/home/centos/hjchen/software/lammps-29Aug2024/src -llammps
# 或者这样也行
# mpicxx lammps_cplusplus_API_ex.cpp -o lmp_cpp_api  -L/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/lammps  -llammps

```

具体的操作，如下：

```sh
(base) [centos@localhost neighborlist_code]$ conda deactivate
[centos@localhost neighborlist_code]$ conda activate neighlist_env
(neighlist_env) [centos@localhost neighborlist_code]$ python -V
Python 3.10.14
(neighlist_env) [centos@localhost neighborlist_code]$ export PATH=/home/centos/hjchen/mpich-4.1/_build/bin:$PATH
(neighlist_env) [centos@localhost neighborlist_code]$ export CPATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$CPATH
(neighlist_env) [centos@localhost neighborlist_code]$ export LD_LIBRARY_PATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$LD_LIBRARY_PATH
(neighlist_env) [centos@localhost neighborlist_code]$ which mpicxx
~/hjchen/mpich-4.1/_build/bin/mpicxx
(neighlist_env) [centos@localhost neighborlist_code]$ which liblammps.so
~/hjchen/software/lammps-29Aug2024/src/liblammps.so
(neighlist_env) [centos@localhost neighborlist_code]$ 
(neighlist_env) [centos@localhost neighborlist_code]$ mpicxx lammps_cplusplus_API_ex.cpp  -o lmp_cpp_api -L/home/centos/hjchen/software/lammps-29Aug2024/src -llammps
(neighlist_env) [centos@localhost neighborlist_code]$ 
(neighlist_env) [centos@localhost neighborlist_code]$ ./lmp_cpp_api 
LAMMPS (29 Aug 2024)
LAMMPS version ID: 20240829
Total wall time: 0:00:00
(neighlist_env) [centos@localhost neighborlist_code]$ 
```

**Q1: 这里有个疑问：为什么我添加了lammps的动态链接库路径，还需要-L它才能编译成功呢?**  

**答**：即使您已经将 LAMMPS 的动态链接库路径添加到 `LD_LIBRARY_PATH` 环境变量中，编译时仍然需要使用 `-L` 选项指定库文件的路径，这是因为：

1. **编译时和运行时的区别**：
   - `LD_LIBRARY_PATH` 环境变量用于运行时动态链接库的搜索路径，它帮助操作系统在执行程序时找到所需的动态链接库。
   - 编译时，编译器和链接器需要知道库文件的位置以解析外部引用。这是通过 `-L` 选项来指定库文件路径，并通过 `-l` 选项指定库的名称来完成的。

2. **静态链接和动态链接**：
   - 编译时链接是指在编译过程中将程序代码与库代码结合起来，生成一个包含所有所需代码的可执行文件。这通常涉及到静态库或已经编译好的动态库的直接复制。
   - 运行时链接（动态链接）是指在程序执行时由操作系统加载所需的库。`LD_LIBRARY_PATH` 用于帮助操作系统找到这些库。

3. **编译器和链接器的需要**：
   - 当使用 `mpicxx` 编译程序时，编译器（在这种情况下是 MPI 编译器包装器）需要知道库文件的确切位置，以便正确链接程序中的函数调用和变量引用。
   - `-L` 选项告诉编译器在哪些目录中查找库文件，而 `-l` 选项指定了要链接的库的名称（不包括前缀 `lib` 和后缀 `.so`）。

4. **环境变量的局限性**：
   - 环境变量 `LD_LIBRARY_PATH` 只在运行时有效，编译器在编译时不会考虑它。
   - 如果编译器在编译时找不到库文件，它无法生成正确的二进制代码来调用库中的函数。

因此，即使您已经设置了 `LD_LIBRARY_PATH`，您仍然需要在编译命令中使用 `-L` 选项来指定库文件的路径，以确保编译器可以找到并正确链接到 LAMMPS 库。

总结一下，`-L` 和 `-l` 选项在编译时用于静态或动态链接，而 `LD_LIBRARY_PATH` 仅在运行时用于动态链接。两者都很重要，但用于不同的目的。

---

**Q2: 具体点，为什么我添加了lammps的动态链接库路径`export LD_LIBRARY_PATH=/home/centos/hjchen/software/lammps-29Aug2024/src:$LD_LIBRARY_PATH`，还需要-L它才能编译成功呢?  `mpicxx lammps_cplusplus_API_ex.cpp -L/home/centos/hjchen/software/lammps-29Aug2024/src -llammps`**

**答**：添加 `LD_LIBRARY_PATH` 环境变量是为了帮助操作系统**在运行时**找到动态链接库，而不是在编译时。编译器和链接器**在编译过程中需要知道库文件的确切位置**，这就是为什么您需要使用 `-L` 选项来指定库文件路径的原因。

当您编译一个程序时，编译器需要确保所有引用的函数和变量都能正确链接到程序中。这包括查找所有依赖的库文件，并将它们与您的程序代码结合起来。在这个阶段，编译器和链接器不会查看 `LD_LIBRARY_PATH`，因为这是一个运行时的环境变量。

以下是详细解释：

1. **编译时链接**：
   - 当您使用 `mpicxx` 编译程序时，编译器需要找到所有头文件（由 `-I` 选项指定）和库文件（由 `-L` 选项指定）。
   - `-L` 选项告诉链接器在哪里查找库文件。例如，`-L/home/centos/hjchen/software/lammps-29Aug2024/src` 告诉链接器在 `/home/centos/hjchen/software/lammps-29Aug2024/src` 目录下查找库文件。
   - `-l` 选项告诉链接器链接到哪个库。例如，`-llammps` 告诉链接器链接到 `liblammps.so`（编译器会寻找 `lib` 前缀和 `.so` 后缀之间的名称）。

2. **运行时链接**：
   - `LD_LIBRARY_PATH` 是一个环境变量，它在程序运行时被操作系统用来查找动态链接库。
   - 当您的程序执行一个动态链接的函数调用时，操作系统会查找 `LD_LIBRARY_PATH` 指定的路径，以找到所需的 `.so` 文件。

3. **为什么需要 `-L`**：
   - 即使您已经将库路径添加到 `LD_LIBRARY_PATH`，编译器在编译时仍然不知道这个库的存在，因为它不会查看 `LD_LIBRARY_PATH`。
   - `-L` 选项是编译时链接所必需的，它告诉编译器库文件的位置。

4. **编译命令**：
   - 您的编译命令 `mpicxx lammps_cpluplus_API_ex.cpp -L/home/centos/hjchen/software/lammps-29Aug2024/src -llammps` 正确地指定了库文件的位置，这样编译器就能找到并链接到 `liblammps.so`。

总结来说，`-L` 选项是为了**编译时**链接，而 `LD_LIBRARY_PATH` 是为了**运行时**链接。两者都很重要，但用途不同。**在编译时，您必须使用 `-L` 选项来指定库文件的路径**。


---
### 补充：asap3库下载
```sh
[centos@localhost src]$ conda activate neighlist_env
(neighlist_env) [centos@localhost neighborlist_code]$ export PATH=/home/centos/hjchen/mpich-4.1/_build/bin:$PATH 
(neighlist_env) [centos@localhost neighborlist_code]$ which mpicxx
~/hjchen/mpich-4.1/_build/bin/mpicxx
(neighlist_env) [centos@localhost neighborlist_code]$ cd adap
(neighlist_env) [centos@localhost asap]$ python setup.py build
...
...
mpicxx -pthread -B /home/centos/anaconda3/envs/neighlist_env/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/centos/anaconda3/envs/neighlist_env/include -fPIC -O2 -isystem /home/centos/anaconda3/envs/neighlist_env/include -fPIC -DASAP_GITHASH=c513f039fb92ecc274d9bfd35bebe0e23d896b69 -UNDEBUG -IBasics -IPotentials -IInterface -IBrenner -ITools -IPTM -IPTM/qcprot -IPTM/voronoi -IParallel -IParallelInterface -IVersionInfo_autogen -I/home/centos/anaconda3/envs/neighlist_env/include/python3.10 -I/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/numpy/core/include -c PTM/alloy_types.cpp -o build/temp.linux-x86_64-cpython-310/PTM/alloy_types.o -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-function -Wno-c++11-compat-deprecated-writable-strings -Wno-unknown-attributes
mpicxx -pthread -B /home/centos/anaconda3/envs/neighlist_env/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/centos/anaconda3/envs/neighlist_env/include -fPIC -O2 -isystem /home/centos/anaconda3/envs/neighlist_env/include -fPIC -DASAP_GITHASH=c513f039fb92ecc274d9bfd35bebe0e23d896b69 -UNDEBUG -IBasics -IPotentials -IInterface -IBrenner -ITools -IPTM -IPTM/qcprot -IPTM/voronoi -IParallel -IParallelInterface -IVersionInfo_autogen -I/home/centos/anaconda3/envs/neighlist_env/include/python3.10 -I/home/centos/anaconda3/envs/neighlist_env/lib/python3.10/site-packages/numpy/core/include -c PTM/canonical.cpp -o build/temp.linux-x86_64-cpython-310/PTM/canonical.o -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-function -Wno-c++11-compat-deprecated-writable-strings -Wno-unknown-attributes
...
...
ardJones.o build/temp.linux-x86_64-cpython-310/Potentials/MetalOxideInterface.o build/temp.linux-x86_64-cpython-310/Potentials/MetalOxideInterface2.o build/temp.linux-x86_64-cpython-310/Potentials/MonteCarloEMT.o build/temp.linux-x86_64-cpython-310/Potentials/Morse.o build/temp.linux-x86_64-cpython-310/Potentials/Potential.o build/temp.linux-x86_64-cpython-310/Potentials/RGL.o build/temp.linux-x86_64-cpython-310/Potentials/RahmanStillingerLemberg.o build/temp.linux-x86_64-cpython-310/Tools/CNA.o build/temp.linux-x86_64-cpython-310/Tools/CoordinationNumbers.o build/temp.linux-x86_64-cpython-310/Tools/FullCNA.o build/temp.linux-x86_64-cpython-310/Tools/GetNeighborList.o build/temp.linux-x86_64-cpython-310/Tools/RawRadialDistribution.o build/temp.linux-x86_64-cpython-310/Tools/SecondaryNeighborLocator.o build/temp.linux-x86_64-cpython-310/VersionInfo_autogen/version_info.o -o build/lib.linux-x86_64-cpython-310/_asap.cpython-310-x86_64-linux-gnu.so
running build_scripts
creating build/scripts-3.10
copying and adjusting scripts/asap-qsub -> build/scripts-3.10
copying and adjusting scripts/asap-sbatch -> build/scripts-3.10
changing mode of build/scripts-3.10/asap-qsub from 664 to 775
changing mode of build/scripts-3.10/asap-sbatch from 664 to 775
```
- 使用git下载asap3：`git clone https://gitlab.com/asap/asap.git`, 然后`cd asap`后，执行`python setup.py build`编译     

- 添加`scripts-3.10`和`PYTHONPATH`到环境变量中（需要根据实际asap安装情况改）  
  - 若在GPU节点环境，则执行：`export PATH=/home/centos/hjchen/GNN_painn/neighborlist_code/asap/build/scripts-3.10:$PATH`。此外，添加`PYTHONPATH`到环境变量中，否则`import asap3`找不到加载模块：`export PYTHONPATH="/home/centos/hjchen/GNN_painn/neighborlist_code/asap/build/lib.linux-x86_64-cpython-310:$PYTHONPATH"`