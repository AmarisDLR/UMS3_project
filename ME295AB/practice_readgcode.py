import re
import time

gfile = "UMgcode/square.gcode"


#with open(gfile) as gcode:
	#print(gfile)
	#for line in gcode:
		#line = line.strip()
		#layerend = re.findall('LAYER:', line)
		#if layerend:
			#layernum = re.findall(r'-?\d+', line)
			#print(layernum[0])

def find_init_temperature(gfile):
	extruder_temp = 0
	bed_temp = 0
	with open(gfile) as gcode:
		for line in gcode:
			#print(line)
			line = line.strip()
			extemp = re.findall(";EXTRUDER_TRAIN.0.INITIAL_TEMPERATURE:", line)
			btemp = re.findall(";BUILD_PLATE.INITIAL_TEMPERATURE:", line)
			if extemp:
				extruder_temp = line.split(":",1)[1]
				print(extruder_temp)
				flag0 = 1
			if btemp:
				bed_temp = line.split(":",1)[1]
				print(bed_temp)
				flag1 = 1 
			if extruder_temp and bed_temp:
				print("here")
				return extruder_temp, bed_temp
				break


#temp0, temp1 = find_init_temperature(gfile)

#print(temp0)

###############################################
#file0 = open(gfile, 'r')
#count = 0
#while True:
	#try:	
		#time.sleep(.0001)
		#count += 1
		#print("\n\n"+str(count)+"\n\n")

		#line = file0.readline()
		#print(line)
		#if not line:
		#	break
	#except(KeyboardInterrupt):
		#print("\nExit program.\n")
		#break


################################################
#times = open("times.txt","w+")

#count = 1
#time1 = 1
#with open(gfile) as gcode:
#	print(gfile)
#	for line in gcode:
#		line = line.strip()
#		timeline = re.findall(';TIME_ELAPSED:', line)
#		if timeline:
#			time2 = re.findall(r'[\d.\d]+', line)
#			time2 = float(time2[0])
#			time_elapsed = time2 - time1
#			times.write(str(time_elapsed)+","+str(count)+"\n")
#			print(time_elapsed)
#			time1 += 1
#			print(count)
#		count += 1

def get_time_elapsed(times_file):
	line = times_file.readline()
	items = line.split(",")
	elapsed_time = re.findall(r'[\d.\d]+', items[0])
	time_line = re.findall(r'[\d.\d]+', items[1])
	elapsed_time = float(elapsed_time[0])
	time_line = int(time_line[0])
	return elapsed_time, time_line

times_file = "times.txt"
linecount = 1
times = open(times_file,"r")

elapsed_time, layerbreak = get_time_elapsed(times)
print(elapsed_time, layerbreak)

while True:
	try:
		time.sleep(.1)
		print(linecount)

		if linecount == layerbreak:
			#time.sleep(float(elapsed_time))
			elapsed_time, layerbreak = get_time_elapsed(times)
			print(elapsed_time, layerbreak)

		#print(elapsed_time, layerbreak)
		#u = input("\n\nget_time_elapsed\n\n")

		linecount += 1

	except(KeyboardInterrupt):
		break
		print("\nExit program.\n")

