# data pre-processing
import pandas as pd
import numpy as np
import scipy
import os
import ase.io.vasp

# data visualization
import matplotlib.pyplot as plt

# materials_project API
from mp_api.client import MPRester
from emmet.core.electronic_structure import BSPathType



def indzero(phfreq):
    s = None
    for i in range(len(phfreq)):
        if abs(phfreq[i]) < 1e-6 or phfreq[i] > 0:
            s = i
            break
    return s



def download_data(df: pd.DataFrame, freq: int = 20, window_length: int = 101, polyorder: int = 3, save_name: str = "../data/downloaded_data.csv"):
    s = []
    for i in np.arange(len(df)):
        try:
            with MPRester("MxyYdMFMXD0vAlGP36qIgFeGmhl7DAY9") as mpr:
                ph_dos = mpr.get_phonon_dos_by_material_id(df.iloc[i]["mp_id"]) #downloading phonon data
                # Smoothing raw spectre using Savitzky-Goley filter of window length 101 and polynomial order 3
                # Используется фильт Савицкого-Голея с шириной окна 101 и степенью полинома 3
                new_dos = scipy.signal.savgol_filter(list(ph_dos.densities), window_length = window_length, polyorder = polyorder) 

                new_freq = list(ph_dos.frequencies)
                new_freq = list(np.ceil(np.array(new_freq)/0.0299792458))
                firstzero = indzero(new_freq)
                new_freq = new_freq[firstzero: 1000 + firstzero: freq]
                new_freq[0] = 0.0
                new_dos = list(new_dos[firstzero: 1000 + firstzero: freq])
                if len(new_freq) <= (1000 / freq):
                    new_freq = new_freq + list(i for i in range(len(new_freq) * freq, 1000 + freq, freq))
                    new_dos = new_dos + list(0 for i in range(len(new_dos) * freq, 1000 + freq, freq))
                a = new_dos / np.max(new_dos)
                new_dos = list(np.where(a >= 0, a, 0.0))

                new_pdos = dict()
                for k, v in ph_dos.get_element_dos().items():
                    firstzero = indzero(np.ceil(v.frequencies / 0.0299792458))
                    v.densities = list(scipy.signal.savgol_filter(list(v.densities), window_length = window_length, polyorder = polyorder))
                    v.densities = v.densities[firstzero:1000 + firstzero:freq]
                    if len(v.densities) < (1000 / freq):
                        v.densities = list(v.densities) + [i for i in np.arange(len(v.densities) * freq, 100, freq)]
                    a = v.densities / np.max(v.densities)
                    new_pdos[str(k)] = list(np.where(a >= 0, a, 0.0))

                df.iloc[i].pdos = new_pdos
                df.iloc[i].phdos = new_dos
                df.iloc[i].phfreq = new_freq
        except:
            s.append(i)
    df.drop(axis = 0, index = s ,inplace = True)
    df.to_csv(save_name, index=False)



def is_phonon_dir(dirname: str):
    """
    Проверяем директорию хранит ли она файл total_dos.dat и POSCAR
    Также проверяется наличие одной из вложенных папок 'phonon' или 'phonopy'
    """
    if os.path.exists(os.path.join(dirname, "total_dos.dat")) and os.path.exists(os.path.join(dirname, "POSCAR")):
        phonon_dir = dirname
    elif os.path.exists(os.path.join(dirname, "phonon")):
        phonon_dir = os.path.join(dirname, "phonon")
    elif os.path.exists(os.path.join(dirname, "phonopy")):
        phonon_dir = os.path.join(dirname, "phonopy")
    else:
        phonon_dir = None

    return phonon_dir



def convert_phdos(phdos, phfreq, freq = 20, window_length = 101, polyorder = 3):
    phfreq = np.array(phfreq)/0.0299792458
    interpolator = scipy.interpolate.interp1d(phfreq, phdos, fill_value = "extrapolate")
    phfreq = list(float(i) for i in range(0, 1001))
    phdos = interpolator(phfreq)
    index = indzero(phfreq)
    phdos = scipy.signal.savgol_filter(phdos, window_length = window_length, polyorder = polyorder)
    phdos = list(np.where(phdos >= 0, phdos, 0.0))
    phfreq = phfreq[index:1001 + index:freq]
    phdos = phdos[index:1001 + index:freq]
    phdos = list(phdos / np.max(phdos))
    if len(phfreq) <= (1000 / freq):
        phfreq = phfreq + list(float(i) for i in range(len(phfreq) * freq, 1000 + freq, freq))
        phdos = phdos + list(0.0 for i in range(len(phdos) * freq, 1000 + freq, freq))
    return phdos, phfreq



def read_phonon(directory: str):
    phonon_dat_file = os.path.join(directory, "total_dos.dat")
    poscar_file = os.path.join(directory, "POSCAR")
    if os.path.exists(phonon_dat_file) and os.path.exists(poscar_file):
        phdos = []
        phfreq = []
        with open(phonon_dat_file) as file:
            data = file.readlines()
            for line in data:
                if not "#" in line:
                    _ = line.strip().split()
                    phfreq.append(eval(_[0]))
                    phdos.append(eval(_[1].strip("\n")))
        new_struct = {}
        structure = ase.io.vasp.read_vasp(poscar_file).todict()
        for k, v in structure.items():
            new_struct[k] = v.tolist()
        structure = new_struct
    else:
        phdos = None
        phfreq = None
        structure = None
    if phdos and phfreq and structure:
        phdos, phfreq = convert_phdos(phdos = phdos, phfreq = phfreq)
        result = pd.Series({"structure": structure, "phfreq": phfreq, "phdos": phdos})
    else:
        result = None
    return result



def read_dir(directory: str):
    resulted_df = pd.DataFrame(columns = ["structure", "phfreq", "phdos"])
    for dirname in os.listdir(directory):
        phonon_dir = is_phonon_dir(dirname = os.path.join(directory, dirname))
        if phonon_dir:
            phonon = read_phonon(phonon_dir)
            if type(phonon) != "NoneType" and type(phonon) != float:
                resulted_df.loc[len(resulted_df)] = phonon
            else:
                raise ValueError
    return resulted_df



def make_sets(df: pd.DataFrame, test_size:float = 0.124, valid_size:float = 0.093):
    train_set = []
    test_set = set()
    valid_set = set()

    while len(test_set)<(round(len(df)*test_size)):
        a = np.random.randint(len(df))
        test_set.add(a)

    while len(valid_set)<(round(len(df)*valid_size)):
        a = np.random.randint(len(df))
        if a not in test_set:
            valid_set.add(a)

    while len(train_set) < (len(df) - len(test_set) - len(valid_set)):
        a = np.random.randint(len(df))
        if a not in test_set and a not in valid_set:
            train_set.append(a)

    with open("../data/idx_train.txt", "w") as file:
        for i in train_set:
            file.write(str(i)+"\n")

    with open("../data/idx_test.txt", "w") as file:
        for i in test_set:
            file.write(str(i)+"\n")

    with open("../data/idx_valid.txt", "w") as file:
        for i in valid_set:
            file.write(str(i)+"\n")