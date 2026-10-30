import runpy, subprocess, sys
subprocess.run("mkdir -p /content/arc && tar -xzf /content/arc.tar.gz -C /content/arc && mkdir -p /content/adapters && tar -xzf /content/adapter.tar.gz -C /content/adapters && pip uninstall -y -q torchao", shell=True, check=True)
sys.argv = ["driver", "confirm"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
