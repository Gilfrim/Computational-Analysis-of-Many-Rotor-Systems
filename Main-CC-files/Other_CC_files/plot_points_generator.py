import subprocess
import numpy as np
import re
import ast

#CHANGE TO FOLDER WHERE FILES ARE LOCATED
base_path = r"/Users/gilfrim/Desktop/Quantum\ Chemistry\ Coop/Main-CC-files"

input_file = base_path + r"\input.txt"
script_file = base_path + r"\Periodic.py"
plots_file = base_path + r"\plot_points.csv"
points = []

#cycles through sites and g values to change in the input file to run the periodic system
for sites in range(49, 50):
    for g in np.arange(0.25, 0.25001, 0.01):
        with open(input_file, "r") as f:
            input_content = f.read()

        #the "(g\s*=\s*)\S+" strings are RegEx
        updated_content = re.sub(r"(g\s*=\s*)\S+", lambda m: f"{m.group(1)}{g}", input_content)
        updated_content = re.sub(r"(sites\s*=\s*)\S+", lambda m: f"{m.group(1)}{sites}", updated_content)

        with open(input_file, "w") as f:
            f.write(updated_content)

        #runs the periodic file and = the printed output
        result = subprocess.run(["python", script_file], capture_output = True, text = True)

        #periodic needs to print these values out otherwise this can't find those values
        energy_match = re.search(r"Energy\s*:\s*(\S+)", result.stdout)
        t_1_max_match = re.search(r"t_1_max\s*=\s*(\S+)", result.stdout)
        t_2_max_match = re.search(r"t_2_max\s*=\s*(\S+)", result.stdout)

        #checks that values were found
        if t_1_max_match and t_2_max_match and energy_match:
            t_1_max = float(t_1_max_match.group(1))
            t_2_max = float(t_2_max_match.group(1))
            energy = float(energy_match.group(1))
            points.append((sites, float(g), t_1_max, t_2_max, energy))
        else:
            print(f"Error: Could not find t_1_max or t_2_max in the output. g: {g}, site: {sites}")

        max_per_y_match = re.search(r"max_per_y\s*=\s*(\[[^\]]*\])", result.stdout)

        if max_per_y_match:
            max_per_y_str = max_per_y_match.group(1)
            try:
                max_per_y = ast.literal_eval(max_per_y_str)
            except (SyntaxError, ValueError):
                print("Could not find max_per_y")
        else:
            print("no max y")

#writes values to output file
with open(plots_file, mode = "w") as out:
    out.write(f"sites, g, t_1_max, t_2-max, energy\n")
    for index in points:
        out.write(f"{index[0]}, {index[1]}, {index[2]}, {index[3]}, {index[4]}\n")

max_y_file = r"C:\Users\Bryan\Desktop\Coop\Coop project\max_per_y.csv"
with open(max_y_file, mode = "w") as max_out:
    max_out.write(f"site, max t_2\n")
    for index, value in enumerate(max_per_y):
        max_out.write(f"{index}, {value}\n")