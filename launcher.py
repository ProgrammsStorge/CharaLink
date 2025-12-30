import os
import sys

if not os.path.exists(f"characters\\{sys.argv[1]}"):
    os.system(f"clim install {sys.argv[1]}")

os.system("endpoint")