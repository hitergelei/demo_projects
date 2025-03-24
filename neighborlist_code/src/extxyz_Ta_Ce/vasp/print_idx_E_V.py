from ase.io import read
import numpy as np
#idx = ["0.86","0.88", "0.90", "0.92", '0.95', '0.98', '1.00', '1.02', '1.05', '1.07', '1.10', '1.12', '1.14']
#idx = ["0.78", "0.80", "0.82", "0.84", "0.86", "0.88", "0.90", "0.92", "0.94", "0.96", "0.98", "1.00", "1.02", "1.04", "1.06", "1.08", "1.10", "1.12", "1.14", "1.16", "1.18", "1.20", "1.22"]
idx = ["%.3f"%i for i in np.arange(0.78, 1.221, 0.005)]

for i in idx:
    #st = read("POSCAR_scal{}.vasp".format(i),  format='vasp')
    st = read("dir_POSCAR_{}/OUTCAR".format(i), format="vasp-out")
    #E = st.get_potential_energy()
    E = st.get_total_energy()
    V = st.get_volume()
    N = st.get_global_number_of_atoms()
    #print('POSCAR_scal{}的  E_per = {:>20}    V_per =  {:>20}'.format(i, E/N, V/N))
    print('{} {:>20}  {:>20}'.format(i, V/N, E/N))  # scal  V  E
    #print(dir(st))
    #exit()
