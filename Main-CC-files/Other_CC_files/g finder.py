import subprocess
import re

base_path = r"C:\Users\Bryan\Desktop\Coop\Coop project"
input_file = base_path + r"\input.txt"
script_file = base_path + r"\Periodic.py"

#Uses binary search to find the value of g that the periodic system diverges
#assumes that g stops working after a specific
#wouldnt work if for example g works for 0.01 to 1 and 5 to 10 cannot be any gaps
low = 0
high = 10

mid_last = 0
while low <= high:
    mid = (high + low) / 2

    with open(input_file, "r") as f_in:
        input_content = f_in.read()

    updated_content = re.sub(r"(g\s*=\s*)\S+", lambda m: f"{m.group(1)}{mid}", input_content)

    with open(input_file, "w") as f_out:
        f_out.write(updated_content)

    result = subprocess.run(["python", script_file], capture_output = True, text = True)

    error_occurred = result.returncode != 0 or result.stderr
    energy_match = re.search(r"Energy\s*:\s*(\S+)", result.stdout)

    if error_occurred or not energy_match:
        high = mid
    else:
        low = mid

    if abs(mid - mid_last) < 1e-10:
        break
    else:
        mid_last = mid
    print(mid)
print(mid)
