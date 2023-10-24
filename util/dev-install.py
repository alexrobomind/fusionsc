import site
import os
import sys

cwd = os.getcwd()

target = None

for i in range(2):
	candidate = os.path.join(cwd, 'src', 'python')
	if os.path.isdir(candidate):
		target = candidate
		break
	
	cwd = os.path.dirname(cwd)

if not target:
	print("Target dir not found")
	sys.exit(1)

try:
	for dir in site.getsitepackages():
		if "site-packages" not in dir:
			continue
			
		print("Trying to write to site packages ", dir)
		with open(os.path.join(dir, "fusionsc.pth"), "w") as f:
			f.write(target)
		
		sys.exit(0)
except PermissionError:
	pass
	
print("Site packages unavailable, falling back to user site packages ", site.getusersitepackages())
with open(os.path.join(site.getusersitepackages(), "fusionsc.pth"), "w") as f:
	f.write(target)