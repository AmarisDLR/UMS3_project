import re
import time

def get_Z(line):
	Z = 0
	items = re.findall('Z', line)
	if items:
		items = line.split("X",1)[1]
		items = items.split()	
		if len(items) > 2: 
			Z = re.findall(r'[\d.\d]+', items[2])
			Z = float(Z[0])
	return Z

def set_time_elapsed(gfile, times_file):
	gfile_read = open(gfile, "r")	
	times = open(times_file,"w+")
	time1 = 0
	count = 1
	Z = 0
	t = True
	while t:
			line = gfile_read.readline()
			if not line:
				t = False
				break
			mesh_line = re.findall('NONMESH', line)
			if mesh_line:
				j = True
				while j:
					line = gfile_read.readline()
					count += 1
					Z = get_Z(line)
					if Z > 0:
						j = False
						break
			timeline = re.findall(';TIME_ELAPSED:', line)
			if timeline:
				time2 = re.findall(r'[\d.\d]+', line)
				time2 = float(time2[0])
				time_elapsed = time2 - time1
				times.write(str(time_elapsed)+","+str(count)+","+str(Z)+"\n")
				time1 = time2
			count += 1



print("\n\n\n\n")
gfile = "gcodeUM/UMS3_Square_nocenter_square.gcode"
print(gfile+"\n")
times_file = "times.txt"
set_time_elapsed(gfile, times_file)




