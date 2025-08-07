import numpy as np
import time
import sys

for i in range(10):
    x = np.random.rand()
    y = np.random.rand()
    z = np.random.rand()

    # Print output block
    print(f"Iteration {i}")
    print(f"  x = {x:.5f}")
    print(f"  y = {y:.5f}")
    print(f"  z = {z:.5f}")
    print(f"  diff(x, y) = {abs(x - y):.5f}")

    time.sleep(0.5)

    # Move the cursor up 5 lines (including the blank line between iterations)
    sys.stdout.write("\033[F" * 5)  # ANSI escape: move cursor up
    sys.stdout.flush()