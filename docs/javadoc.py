from jinja2 import Environment, FileSystemLoader

import javalang
import os
import sys
import traceback

javaPath = sys.argv[1]
rstPath = sys.argv[2]

packages = {}

for folder, subFolder, files in os.walk(javaPath):
	for file in files:
		try:
			absPath = os.path.abspath(os.path.join(folder, file))
			print("Parsing", absPath)
			
			with open(absPath, 'r') as f:
				contents = f.read()
			
			translationUnit = javalang.parse.parse(contents)
			
			package = translationUnit.package.name
			
			if package not in packages:
				packages[package] = {}
			
			for declaration in translationUnit.types:
				packages[package][declaration.name] = declaration
		except Exception as e:
			traceback.print_exc()

env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.realpath(__file__))))
packageTemplate = env.get_template('package.rst.tpl')
classTemplate = env.get_template('class.rst.tpl')

def handleType(fullName, cls):
	renderOutput = classTemplate.render(packageName = name, fullName = fullName, cls = cls, javalang = javalang, isinstance = isinstance, len = len)
	
	with open(os.path.join(rstPath, fullName + ".class.rst"), "w") as f:
		f.write(renderOutput)
	
	for child in cls.body:
		if isinstance(child, javalang.tree.TypeDeclaration):
			handleType(fullName + "." + child.name, child)

for name, classes in packages.items():
	renderOutput = packageTemplate.render(packageName = name, members = classes, javalang = javalang, isinstance = isinstance, len = len)
	
	with open(os.path.join(rstPath, name + ".package.rst"), "w") as f:
		f.write(renderOutput)
	
	for clsName, cls in classes.items():
		handleType(name + "." + clsName, cls)