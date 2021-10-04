## This program will capture an image after every line of the program. 

import os
import time
import calendar
import re
import paramiko
import cv2
from tqdm import tqdm

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
				return float(extruder_temp), float(bed_temp)
				break

def set_temperature(extruder_temp, bed_temp, remote_connection, start):
		
	remote_connection.send("select printer printer/head/0/slot/0 \n")
	remote_connection.send("set pre_tune_target_temperature "+ str(extruder_temp)+" \n")	
	time.sleep(2)
	remote_connection.send("select printer printer/bed \n")
	time.sleep(2)
	remote_connection.send("set pre_tune_target_temperature "+ str(bed_temp)+" \n")
	if start ==1:	
		print("\nTarget Extruder Temperature: " + str(extruder_temp) + " F\n")
		print("Target Bed Temperature: " + str(bed_temp) + " F\n")
		remote_connection.send("sendgcode G0 E2\n")
		print("\n\nStart heating\n\n")
		check_temperature(extruder_temp, bed_temp,remote_connection)
		print("\n\nFinished heating\n\n")
	if start == 0:
		print("\nTarget Extruder Temperature: " + str(extruder_temp) + " F\n")
		print("Target Bed Temperature: " + str(bed_temp) + " F\n")

def check_temperature(extruder_temp, bed_temp,remote_connection):
	t = 1
	count = 0
	remote_connection.recv(9999)
	while t:
		count += 1
		remote_connection.send("select printer printer/bed\n")
		remote_connection.recv(9999)
		remote_connection.send("get current_temperature \n")
		time.sleep(1.5)
		out = str(remote_connection.recv(9999))
		current_bed_temp = re.findall(r'[\d.\d]+', out) # current bed temperature
		if current_bed_temp:
			current_bed_temp = float(current_bed_temp[0])

		remote_connection.send("select printer printer/head/0/slot/0 \n")
		remote_connection.recv(9999)
		remote_connection.send("get current_temperature \n")
		time.sleep(1.5)
		out = str(remote_connection.recv(9999))
		current_ex_temp = re.findall(r'[\d.\d]+', out) # current extruder temperature
		if current_ex_temp:
			current_ex_temp = float(current_ex_temp[0])
		if count%2 == 0:		
			print("*")
		if  current_bed_temp>=bed_temp-0.5 and current_ex_temp>=extruder_temp-0.5:
			print("\nCurrent bed temperature "+str(current_bed_temp)+"\n")
			print("\nCurrent extruder temperature "+str(current_ex_temp)+"\n")			
			t = 0
			break
		if  count > 600:
			t = 0
			break

def zero_bed(gfile, offset, remote_connection): 
	with open(gfile) as gcode:
		for i, line in enumerate(gcode):
			if i ==30:
				time.sleep(1)
				remote_connection.send("select printer printer\n")
				remote_connection.send("set max_speed_z 15\n")
				remote_connection.send("sendgcode G0 Z"+offset+"\n")
				remote_connection.send("sendgcode G92 Z0\n")
				remote_connection.send("set max_speed_z 40\n")
				out = remote_connection.recv(9999)

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

def get_time_elapsed(times_file):
	line = times_file.readline()

	if not line: ### End of File ###
		return -1,-1
		
	items = line.split(",")
	elapsed_time = re.findall(r'[\d.\d]+', items[0])
	time_line = re.findall(r'[\d.\d]+', items[1])
	Z = re.findall(r'[\d.\d]+', items[2])
	elapsed_time = float(elapsed_time[0])
	time_line = int(time_line[0])
	Z = float(Z[0])
	return elapsed_time, time_line, Z

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

def get_XY(line):
	X = 0
	Y = 0
	items = line.split("X",1)[1]
	items = items.split()
	if len(items) > 1:	
		X = re.findall(r'[\d.\d]+', items[0])
		Y = re.findall(r'[\d.\d]+', items[1])
		X = float(X[0])
		Y = float(Y[0])
		
	return X,Y

def check_position(X,Y,Z,remote_connection, webcam, layerbreak):
	t = 1
	count = 0
	while t:
		count += 1
		time.sleep(0.001)
		remote_connection.send("sendgcode M114\n")
		out = remote_connection.recv(9999)
		out = str(out)
		xyz_str = re.search(':(.+?)E',out)
		if xyz_str:
			positions = xyz_str.group(1)
			xyz_items = positions.split(':')
			if len(xyz_items) == 3:
				xc = re.findall(r'[\d.\d]+', xyz_items[0])
				yc = re.findall(r'[\d.\d]+', xyz_items[1])
				zc = re.findall(r'[\d.\d]+', xyz_items[2])
				xc = float(xc[0])
				yc = float(yc[0])
				zc = float(zc[0])
				if xc == X and yc == Y and zc == Z:
					video_capture(webcam, layerbreak)
					t = 0
					return 0
					break
			if count > 100:
				t = 0
				return 1
				break

def capture_img(camera, frame, layerbreak):
	cam_fps = camera.get(cv2.CAP_PROP_FPS)
	print('Capture Image at %.2f FPS.' %cam_fps)
	ts = calendar.timegm(time.gmtime())
	imfile = "defects/"+str(ts)+"_"+str(layerbreak)+'img.jpg'
	print(imfile)
	cv2.imwrite(filename=imfile, img=frame)
	print("Image saved!")

def video_capture(webcam, layerbreak):
	count = 1
	t = True
	while t:
		try:
			time.sleep(0.001)
			count += 1
			check, frame = webcam.read()
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if count == 450: 
				capture_img(webcam, frame, layerbreak)
				t = False
				time.sleep(2)
				break
		except(KeyboardInterrupt):
			cv2.destroyAllWindows()
			break

def print_initial_lines(remote_connection):
	remote_connection.send("sendgcode G0 F600 X0 Z0.24\n")
	remote_connection.send("sendgcode G0 E10\n")
	remote_connection.send("sendgcode G92 E0\n")
	remote_connection.send("sendgcode G0 F1500 Y180\n")
	remote_connection.send("sendgcode G0 Y50 E1\n")
	remote_connection.send("sendgcode G0 X5\n")
	remote_connection.send("sendgcode G0 Y180 E0\n")
	remote_connection.send("sendgcode G0 X10\n")
	remote_connection.send("sendgcode G0 F3000 Y50\n")
	remote_connection.send("sendgcode G92 E0\n")
	time.sleep(2) 

def adjust_extruder(remote_connection, flag):
	remote_connection.send("sendgcode G91\n")	## Set Relative Positioning
	if flag == 1:
		## Retract extruder
		remote_connection.send("sendgcode G0 F5000 E-5\n")
		time.sleep(2)

	remote_connection.send("sendgcode G90\n")	## Set Absolute Positioning
	
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
remote_connection.send("\n")
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G28 \n")
out = remote_connection.recv(9999)
print(out)
################  Get Times Btwn Layers  ##################
print("\n\n\nn")
gfile = "gcodeUM/UMS3_random12_30infill_triangles.gcode" #input("Gcode file: <file.gcode> \n")
print(gfile+"\n")
times_file = "times.txt"
set_time_elapsed(gfile, times_file)

################  Get & Send Temperature  ##################

extruder_temp, bed_temp = find_init_temperature(gfile)
set_temperature(extruder_temp, bed_temp, remote_connection,1)

################  Start Camera  ##################
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)#640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)#480)
webcam.set(cv2.CAP_PROP_FPS, 25)#30)

################  Start Printing  ##################
print("\n\nStart printing from file.\n\n")

z_offset = str(4.2) #input("\nEnter height to zero bed: ")

gfile_print = open(gfile, "r")
times = open(times_file,"r")
linecount = 1

elapsed_time0 = X_line = Y_line = Z_line = 0
elapsed_time, layerbreak, Z = get_time_elapsed(times)
time.sleep(2)

goal_X = 10
goal_Z = Z+1
################  Print Loop  ##################
while True:
	try:

		line = gfile_print.readline()
		check, frame = webcam.read()
		cv2.imshow("Capturing", frame)
		key = cv2.waitKey(1)

		if not line: ### End of File ###
			
			print("\nFinished printing. \n")
			set_temperature(0, 0, remote_connection,0)
			time.sleep(1)
			remote_connection.send("sendgcode G0 F5500 X0 Y150\n")
			time.sleep(1)
			remote_connection.send("sendgcode G28 Z\n")
			command = input("\nConfirm piece is removed from print bed by hitting 'enter'. \n")
			remote_connection.send("sendgcode G28 Z\n")
			ssh.close()
			cv2.destroyAllWindows()
			break

		else: #### Print g-code ####
			time.sleep(0.001)
			print("Line Sent to Printer: "+str(linecount)+" out of "+str(layerbreak))
			remote_connection.send("sendgcode "+line+"\n")

			if linecount == 33: #### Zero Bed Offset ####
				zero_bed(gfile,z_offset,remote_connection)
				print_initial_lines(remote_connection)
				adjust_extruder(remote_connection, 1)
				time.sleep(2)

			if linecount == layerbreak-1:		
				X_line,Y_line = get_XY(line)
				if X_line > 0 and Y_line > 0:
					X = X_line
					Y = Y_line

			if linecount == layerbreak: ## Take image
				print("\n\nLayer: " + str(linecount) +"\n\n")

				## Retract extruder
				set_temperature(extruder_temp-25, bed_temp, remote_connection,2)
				adjust_extruder(remote_connection, 1)

				## Position for camera capture
				goal_Z = Z+1
				goal_Y = 125+(linecount%2)/10 #goal_Y = 130+(linecount%2)/10
				remote_connection.send("sendgcode M400")
				out = remote_connection.recv(9999)
				remote_connection.send("sendgcode G0 F7000 X0 Y150\n")
				remote_connection.send("sendgcode G0 Z"+str(goal_Z)+"\n")

				## Sleep for estimated time for layer			
				for t in tqdm(range(int(elapsed_time)), desc = "Print Progress"):				
					time.sleep(.90)
				remote_connection.send("sendgcode G0 F5000 X"+str(goal_X)+" Y"+str(goal_Y)+" Z"+str(goal_Z)+"\n")
				
				i = 1
				while i == 1:
					i = check_position(goal_X,goal_Y,goal_Z,remote_connection, webcam, layerbreak)
				
				## Position to resume printing
				set_temperature(extruder_temp, bed_temp, remote_connection,2)
				## Adjust_extruder(remote_connection, 0)
				gpositionXY = "X"+str(X)+" Y"+str(Y)
				gpositionZ = " Z"+str(Z)
				## Return to last position
				remote_connection.send("sendgcode G0 "+gpositionZ+"\n")	
				remote_connection.send("sendgcode G0 "+gpositionXY+"\n")	
				time.sleep(2)

				## Update layerbreak and time_break
				elapsed_time, layerbreak, Z = get_time_elapsed(times)

			linecount += 1


	#### Escape/Close Program ####
	except(KeyboardInterrupt):
		print("\nExit program.\n")
		set_temperature(0, 0, remote_connection,0)
		time.sleep(1)
		remote_connection.send("sendgcode G0 F5500 X0 Y150\n")
		time.sleep(1)		
		remote_connection.send("sendgcode G28 Z\n")
		cv2.destroyAllWindows()
		break

set_temperature(0, 0, remote_connection,0)
ssh.close()
#os.remove(times_file)
cv2.destroyAllWindows()
