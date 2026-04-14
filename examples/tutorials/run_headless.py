import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib import

import matplotlib.pyplot as plt
import runpy
import sys

plt.show = lambda *args, **kwargs: None

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        plt.close("all")