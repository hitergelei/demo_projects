from ase.io import read

idx = ["0.86", "0.90", '0.95', '0.98', '1.00', '1.02', '1.05', '1.10', '1.12', '1.14', '1.16']
for i in idx:
    st = read("POSCAR_scal{}.vasp".format(i),  format='vasp')
    #st = read("dir_POSCAR_E_V.{}.vasp/OUTCAR".format(i), format="vasp-out")
    #E = st.get_potential_energy()
    #E = st.get_total_energy()
    V = st.get_volume()
    N = st.get_global_number_of_atoms()
    print('POSCAR_scal{}.vasp的   V_per =  {:>20}'.format(i, V/N))
    #print(dir(st))
    #exit()
