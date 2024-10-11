import os
import subprocess
import platform

os_name = platform.system().lower()
pip_name = "pip3"
python_name = "python3"

print("ADEF: Installation:")
if "windows" == os_name:
    pip_name = pip_name[:-1]
    python_name = python_name[:-1]

print("ADEF: Installing Python Package Requirements:")
# Perform installation and patching process for ADEF framework
subprocess.run([pip_name, "install", "-r", "requirements.txt"])
os.chdir("./patch")
print("ADEF: Applying Python Patches:")
subprocess.run([python_name, "patch.py"])
# subprocess.run(["cd .."])
os.chdir("../")

print("ADEF: Installation Completed Successfully")