import os
import re
import time
import csv

file_path_dmrg = r"/Users/gilfrim/Desktop/Quantum\ Chemistry\ Coop"
file_path_input = file_path_dmrg + r"\input_quick.txt"
file_path_energy = file_path_dmrg + r"\energy.txt"

#runs the dmrg for various sites and basis sizes
sites_start = 2
sites_end = 2

states_start = 5
states_end = 5

m_start = int((states_start - 1) / 2)
m_end = int((states_end + 1) / 2)

#cycles through amount of sites and basis size
for site_new in range(sites_start, sites_end + 1):
    for m_new in range(m_start, m_end):
        if site_new %2 == 0:
            bond_new = int(site_new / 2)
        else:
            bond_new = int(site_new / 2) + 1
        print(f"site: {site_new}")
        print(f"m: {m_new}\n")
        with open(file_path_input, "r", encoding = "utf-8") as input_file:
            input_content = input_file.read()
            #updates values for the dmrg input file
            updated_content = re.sub(r"(m=\s*)\S+", rf"\g<1>{m_new}", input_content)
            updated_content = re.sub(r"(Nsites=\s*)\S+", rf"\g<1>{site_new}", updated_content)
            updated_content = re.sub(r"(mbond=\s*)\S+", rf"\g<1>{bond_new}", updated_content)

        with open(file_path_input, mode = "w") as input_file:
            input_file.write(updated_content)

        #runs the dmrg file
        os.system(f'start /wait cmd /c "cd /d {file_path_dmrg} && julia --project=. main.jl"')

        folder_path = r"/Users/gilfrim/Desktop/Quantum\ Chemistry\ Coop"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = f"dmrg_{site_new}_{m_new}.csv"
        energy_csv_file_path = os.path.join(folder_path, file_name)

        with open(energy_csv_file_path, mode = "w", newline='') as dmrg_csv_file, \
            open(file_path_energy, mode = "r") as energy_file:
            dmrg_csv_file.write(f"g, energy\n")

            for _ in range(5):
                next(energy_file)

            for line_energy in energy_file:

                line_energy = line_energy.strip()
                g, energy = line_energy.split()

                dmrg_csv_file.write(f"{g}, {energy}\n")


