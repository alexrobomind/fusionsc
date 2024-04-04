import sys
import subprocess
import pathlib

pyExe = sys.executable
mainDir = pathlib.Path(__file__) / ".." / ".."
mainDir = mainDir.resolve()

coverage = [pyExe, "-m", "coverage"]

subprocess.run(coverage + ["run", "-m", "pytest", str(mainDir / "src" / "python" / "tests")] + sys.argv[1:])
subprocess.run(coverage + ["html"])