# Colab kernel cell: re-run the driver WITHOUT re-extracting the tarball (keeps hot-fixed files).
import runpy
import sys

sys.argv = ["driver", "baseline"]
runpy.run_path("/content/arc/colab/driver.py", run_name="__main__")
