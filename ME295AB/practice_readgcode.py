import re

gfile = "spacer.gcode"


with open(gfile) as gcode:
	print(gfile)
	for line in gcode:
		line = line.strip()
		layerend = re.findall('LAYER:', line)
		if layerend:
			layernum = re.findall(r'-?\d+', line)
			print(layernum[0])
