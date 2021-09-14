import os
import time
import calendar
import re
import paramiko


def find_init_temperature(gfile):
# First, find the initial temperature
# M109 and M104 do not work in SSH griffin
	extruder_temp = 0
	bed_temp = 0
	with open(gfile) as gcode:
		for line in gcode:
			line = line.strip()
			extemp = re.findall(";EXTRUDER_TRAIN.0.INITIAL_TEMPERATURE:", line)
			btemp = re.findall(";BUILD_PLATE.INITIAL_TEMPERATURE:", line)
			if extemp:
				extruder_temp = line.split(":",1)[1]
			if btemp:
				bed_temp = line.split(":",1)[1]
			if extruder_temp and bed_temp:
				return extruder_temp, bed_temp
				break

def set_temperature(extruder_temp, bed_temp, remote_connection):
	print("\nInitial Extruder Temperature: " + str(extruder_temp) + " F\n")		
	out = remote_connection.recv(9999)	
	remote_connection.send("select printer printer/head/0/slot/0 \n")
	#remote_connection.send("set pre_tune_target_temperature "+ str(extruder_temp)+" \n")	
	time.sleep(2)
	remote_connection.send("get current_temperature \n")
	time.sleep(2)
	out = str(remote_connection.recv(9999))
	out = out[-16:-1]
	current_ex_temp = re.findall(r'[\d.\d]+', out) # current extruder temperature ex. 29.4 --> ['29', '4']
	print(current_ex_temp)
	remote_connection.send("select printer printer/bed \n")
	time.sleep(2)

	#remote_connection.send("set pre_tune_target_temperature "+ str(bed_temp)+" \n")

	print("\n\nStart heating\n\n")
	#time.sleep(230)
	input("\n\nstop\n\n")

def zero_bed(gfile, offset, remote_connection): 
	out = remote_connection.recv(9999)
	print(out)

	with open(gfile) as gcode:
		for i, line in enumerate(gcode):
			if i ==30:
				remote_connection.send("select printer printer\n")
				remote_connection.send("set max_speed_z 15\n")
				remote_connection.send("sendgcode G0 Z"+offset+"\n")
				remote_connection.send("sendgcode G92 Z0\n")	
				remote_connection.send("set max_speed_z 40\n")
				break

def set_time_elapsed(gfile, times_file):
	times = open(times_file,"w+")
	time1 = 0
	count = 1
	with open(gfile) as gcode:
		print(gfile)
		for line in gcode:
			line = line.strip()
			timeline = re.findall(';TIME_ELAPSED:', line)
			if timeline:
				time2 = re.findall(r'[\d.\d]+', line)
				time2 = float(time2[0])
				time_elapsed = time2 - time1
				times.write(str(time_elapsed)+","+str(count)+"\n")
			count += 1

def get_time_elapsed(times_file):
	line = times_file.readline()
	items = line.split(",")
	elapsed_time = re.findall(r'[\d.\d]+', items[0])
	time_line = re.findall(r'[\d.\d]+', items[1])
	elapsed_time = float(elapsed_time[0])
	time_line = int(time_line[0])
	return elapsed_time, time_line

def get_Z(line):
	items = line.split("X",1)[1]
	items = items.split()
	Z = re.findall(r'[\d.\d]+', items[2])
	Z = float(Z[0])
	return Z

def get_XY(line):
	items = line.split("X",1)[1]
	items = items.split()
	X = re.findall(r'[\d.\d]+', items[0])
	Y = re.findall(r'[\d.\d]+', items[1])
	X = float(X[0])
	Y = float(Y[0])
	return X,Y

################  Log in to SSH  ##################
ip_address = "10.1.10.203"
username = "ultimaker"
password = "ultimaker"
ssh = paramiko.SSHClient()
# Load SSH host keys.
ssh.load_system_host_keys()

# Add SSH host key when missing.
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip_address, username=username, password=password, look_for_keys=False, port=22)
remote_connection = ssh.invoke_shell()

# send out command to begin shell program
remote_connection.send("help \n")
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G28 \n")
out = remote_connection.recv(9999)
print(out)

################  Get & Send Temperature  ##################

print("\n\n\n\n")

gfile = "UMgcode/square.gcode" #input("Name of gcode file to print: <file.gcode> \n")
times_file = "times.txt"
print(gfile+"\n")
extruder_temp, bed_temp = find_init_temperature(gfile)
set_temperature(extruder_temp, bed_temp, remote_connection)

################  Get Times Btwn Layers  ##################

set_time_elapsed(gfile, times_file)

################ Start Printing  ##################
print("\n\nStart printing from file.\n\n")

gfile_print = open(gfile, "r")
times = open(times_file,"r")
linecount = 1

elapsed_time, layerbreak = get_time_elapsed(times)
print(elapsed_time, layerbreak)
u = input("\n\nget_time_elapsed\n\n")

while True:
	try:

		line = gfile_print.readline()

		if not line: 
			print("\nFinished printing. \n")
			remote_connection.send("sendgcode G28 \n")
			remote_connection.send("sendgcode G0 Z60\n")

			command = input("\nConfirm piece is removed from print bed by hitting 'enter'. \n")
			remote_connection.send("set pre_tune_target_temperature "+ "0 "+" \n")
			remote_connection.send("sendgcode G28 \n")
			ssh.close()
			break

		else: #### Print g-code ####
			time.sleep(0.01)
			print(linecount,layerbreak)
			#print("\n\n"+str(linecount)+"\n\n")
			#print(line)
			#remote_connection.send("sendgcode " +line +" \n")
			#out = remote_connection.recv(9999)

			if linecount == 30: #### Zero Bed Offset ####
				print(line)
				z_offset = input("\nEnter height to zero bed: ")
				zero_bed(gfile,z_offset,remote_connection)

			if linecount == layerbreak-2:
				print("\n\nGet Z\n\n")
				Z = get_Z(line)
				print(Z)
			
			if linecount == layerbreak-1:
				print("\n\nGet X and Y\n\n")
				X,Y = get_XY(line)
				print(X,Y)

			if linecount == layerbreak: ## Take image
				print("\n\nLine " + str(linecount) + ": Pause " + str(elapsed_time) + " s\n\n")
				time.sleep(float(elapsed_time))

				#update layerbreak and time_break
				elapsed_time, layerbreak = get_time_elapsed(times)

			linecount += 1


	#### Escape/Close Program ####
	except(KeyboardInterrupt):
		print("\nExit program.\n")
		remote_connection.send("set pre_tune_target_temperature "+ "0 "+" \n")
		remote_connection.send("get current temperature \n")
		remote_connection.send("sendgcode G28 \n")
		os.remove(times_file)
		ssh.close()
		break
ssh.close()
os.remove(times_file)
