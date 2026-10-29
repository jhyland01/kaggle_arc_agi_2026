# Colab kernel cell: unpack the project and run the driver in the foreground (keeps the kernel busy).
import runpy
import subprocess
import sys

subprocess.run("mkdir -p /content/arc && tar -xzf /content/arc.tar.gz -C /content/arc", shell=True, check=True)
sys.argv = ["driver", "baseline"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
