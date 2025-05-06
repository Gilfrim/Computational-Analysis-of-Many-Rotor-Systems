import subprocess
import numpy as np
import re
input_file = "input.txt"
output_file = "current.txt"
script_file = "Non_Periodic.py"

#range of g values
g_values = np.arange(0.05, 0.65, 0.05)

for g in g_values:
    # Read input.txt content
    with open(input_file, "r") as f:
        input_content = f.read()

    # Replace only the g value in the
    updated_content = re.sub(r"(g\s*=\s*)\d+\.\d+", rf"\g<1>{g:.2f}", input_content)

    # Write updated content back to input.txt
    with open(input_file, "w") as f:
        f.write(updated_content)

    # Run script.py and capture output
    result = subprocess.run(["python", script_file], capture_output=True, text=True)

    # Extract the last "Energy: <value>" from the output
    energy_values = re.findall(r"Energy:\s*(-?\d+\.\d+)", result.stdout)
    last_energy = energy_values[-1] if energy_values else "N/A"

    # Append final energy value to output.txt
    with open(output_file, "a") as f:
        f.write(f"{last_energy}\n\n\n\n\n")

    #print(f"Run for g = {g:.2f} completed. Last Energy = {last_energy}")