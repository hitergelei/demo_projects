import lammps
from mpi4py import MPI
import subprocess
import sys

from lammps import lammps

# 获取Python版本
python_version = sys.version

print(f"Python version: {python_version}")


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