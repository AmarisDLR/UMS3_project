### This program will take images at evenly spaced breakpoints. Using modulus (%) to
### specify the spacing of the breakpoints.
### This program is geared to the UMS3, which does not use serial communition. Instead
### SSH communication is used to pass on g-code. The g-code file used for the bases of 
### this code is uses g-code flavor Grifffin and does not place the origin at center.
### Using Ultimaker Cura, the origin is placed in the lower left corner of the printer
### bed (default origin).

import os
import time
import re
import paramiko
import cv2
from stl_visualization import *

def find_init_temperature(gfile):
### First, find the initial temperature
### M109 and M104 do not work in SSH griffin
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
	if start ==1:
		print("\nTarget Extruder Temperature: " + str(extruder_temp) + " F\n")
		print("Target Bed Temperature: " + str(bed_temp) + " F\n")
		print("\n\nStart heating\n\n")		
		remote_connection.send("select printer printer/bed \n")
		remote_connection.send("set pre_tune_target_temperature "+ str(bed_temp)+" \n")
		time.sleep(3)
		remote_connection.send("select printer printer/head/0/slot/0 \n")
		remote_connection.send("set pre_tune_target_temperature "+ str(extruder_temp-2)+" \n")	
		time.sleep(2)
		check_temperature(extruder_temp-2, bed_temp,remote_connection)
		print("\n\nFinished heating\n\n")
	if start == 0:
		remote_connection.send("select printer printer/head/0/slot/0 \n")
		remote_connection.send("set pre_tune_target_temperature "+ str(extruder_temp)+" \n")	
		time.sleep(2)
		remote_connection.send("select printer printer/bed \n")
		time.sleep(2)
		remote_connection.send("set pre_tune_target_temperature "+ str(bed_temp)+" \n")
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
		current_bed_temp = re.findall(r'[\d.\d]+', out) ### current bed temperature
		if current_bed_temp:
			current_bed_temp = float(current_bed_temp[0])

		remote_connection.send("select printer printer/head/0/slot/0 \n")
		remote_connection.recv(9999)
		remote_connection.send("get current_temperature \n")
		time.sleep(1.5)
		out = str(remote_connection.recv(9999))
		current_ex_temp = re.findall(r'[\d.\d]+', out) ### current extruder temperature
		if current_ex_temp:
			current_ex_temp = float(current_ex_temp[0])
			if count%2 == 0:		
				print("*",end="",flush=True)
			if  current_bed_temp>=bed_temp-0.5 and current_ex_temp>=extruder_temp-0.5:			
				#print("\nCurrent bed temperature "+str(current_bed_temp)+"\n")
				#print("\nCurrent extruder temperature "+str(current_ex_temp)+"\n")			
				t = 0
				break
		'''if  count > 600:
			t = 0
			break'''

def zero_bed(gfile, offset, remote_connection): 
	with open(gfile) as gcode:
		for i, line in enumerate(gcode):
			if i ==30:
				time.sleep(1)
				remote_connection.send("select printer printer\n")
				remote_connection.send("set max_speed_z 15\n")
				remote_connection.send("sendgcode G0 F450 Z"+offset+"\n")
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
				global final_line
				final_line = count-1
				t = False
				times.write(str(time_elapsed)+","+str(final_line)+","+str(Z+25)+"\n")
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
	if not line: ### if 'End of File' ###
		return -1,-1,-1
		
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
	if re.findall("X",line):
		items = line.split("X",1)[1]
		items = items.split()
		if len(items) > 1:	
			X = re.findall(r'[\d.\d]+', items[0])
			Y = re.findall(r'[\d.\d]+', items[1])
			X = float(X[0])
			Y = float(Y[0])
	return X,Y

def check_position(X,Y,Z,remote_connection,layerbreak):
	count = 0
	X = round(X,2)
	Y = round(Y,2)
	Z = round(Z,2)

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
			x_compare = xc-0.01 <= X <= xc+0.01
			y_compare = yc-0.01 <= Y <= yc+0.01
			z_compare = zc-0.01 <= Z <= zc+0.01
			if x_compare and y_compare and z_compare:
				return 1
	else:
		return 0

def capture_img(img_size_x, img_size_y, gfile_name, gcode_file, camera, frame, layerbreak):
	cam_fps = camera.get(cv2.CAP_PROP_FPS)
	print('Capture Image at %.2f FPS.' %cam_fps)
	ts = time.strftime("%Y%m%d%H%M")
	imfile = 'database/'+str(ts)+'_'+gfile_name+'_'+str(layerbreak)+'.jpg'
	print(imfile)
	cv2.imwrite(filename=imfile, img=frame)
	print("Image saved!")
	im_projection = 'database/'+str(ts)+'_'+'3DProjection_'+gfile_name+'_'+str(layerbreak)+'.jpg'
	gcode_overlay(img_size_x, img_size_y, gcode_file, imfile, layerbreak, im_projection)


def video_capture(img_size_x, img_size_y, gfile_name, gcode_file, webcam, layerbreak):
	count = 1
	t = True
	while t:
		try:
			time.sleep(0.001)
			count += 1
			check, frame = webcam.read()
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if count == 450: ### Give time for camera with autofocus to focus
					 ### & allow time for user to view overhead video
				capture_img(img_size_x, img_size_y, gfile_name, gcode_file, webcam, frame, layerbreak)
				t = False
				time.sleep(2)
				break
		except(KeyboardInterrupt):
			cv2.destroyAllWindows()
			break

def print_initial_lines(remote_connection): ### Purge line ###
	remote_connection.send("sendgcode G0 F600 X0 Z0.24\n")
	remote_connection.send("sendgcode G0 E11\n")
	time.sleep(3)
	set_temperature(extruder_temp, bed_temp, remote_connection,2)
	remote_connection.send("sendgcode G92 E0\n")
	remote_connection.send("sendgcode G0 F1050 Y180\n")
	remote_connection.send("sendgcode G0 Y50 E1\n")
	remote_connection.send("sendgcode G0 F3000 X5 \n")
	remote_connection.send("sendgcode G0 Y100 E-5 \n")

def adjust_extruder(remote_connection, amount):
	remote_connection.send("sendgcode G91\n")	### Set Relative Positioning
	### Retract extruder
	remote_connection.send("sendgcode G0 F8000 E"+str(amount)+"\n")
	time.sleep(2)
	remote_connection.send("sendgcode G90\n")	### Set Absolute Positioning

def adjust_extrusion_amount(line,alt_amount):
	Ennn = line.split('E')
	if Ennn:
		Ennn = Ennn[-1]
		Ennn_alt = float(Ennn) + alt_amount
		line = line.replace("E"+Ennn,"E"+str(Ennn_alt))
	return line

def adjust_feedrate_amount(line,alt_factor):
	if line[0] == 'G':
		Fnnn_split = line.split('F')
		if Fnnn_split:
			Fnnn_split = Fnnn_split[-1]
			Fnnn = re.findall(r'[\d.\d]+', Fnnn_split)
			Fnnn_alt = float(Fnnn[0]) * alt_factor
			line = line.replace("F"+Fnnn[0],"F"+str(Fnnn_alt))
	return line

def adjust_coolingfan_speed(remote_connection,PWM):
	command = "sendgcode M106 S"+str(PWM)
	remote_connection.send(command+"\n")

################  Log in to SSH  ##################
ip_address = "10.1.10.203"
username = "ultimaker"
password = "ultimaker"
ssh = paramiko.SSHClient()
### Load SSH host keys.
ssh.load_system_host_keys()
### Add SSH host key when missing.
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip_address, username=username, password=password, look_for_keys=False, port=22)
remote_connection = ssh.invoke_shell()
### send out command to begin shell program
remote_connection.send("\n")
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G92 E0\n")
remote_connection.send("sendgcode G28 \n")
out = remote_connection.recv(9999)
print(out)

################  Get Times Btwn Layers  ##################
print("\n\n\n")
gfile_name = "UMS3_random39_7infill_lines"
gfile = "gcodeUM/"+gfile_name+".gcode" #input("Gcode file: <file.gcode> \n")
gcode_file = "gcodeUM/UMS3_random39_7infill_lines_MARLIN.gcode"
stl_file = 'gcodeUM/UMS3_random39_block.stl'
print(gfile+"\n")
times_file = "times.txt"
set_time_elapsed(gfile, times_file)

################  Get & Send Temperature  ##################
extruder_temp, bed_temp = find_init_temperature(gfile)
set_temperature(extruder_temp, bed_temp, remote_connection,1)

################  Start Camera  ##################
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
img_size_x = 1600
img_size_y = 1200
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, img_size_x)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size_y)
#webcam.set(cv2.CAP_PROP_FPS, 15)

################  Start Printing  ##################
print("\n\nStart printing from file.\n\n")

z_offset = str(4.22) # z_offset = input("\nEnter height to zero bed: ")

gfile_print = open(gfile, "r")
times = open(times_file,"r")
linecount = layercount = 1

elapsed_time0 = X_line = Y_line = Z_line = 0
elapsed_time, layerbreak, Z = get_time_elapsed(times)
timepause = elapsed_time
time.sleep(2)

goal_X = 10
goal_Z = Z+2

fr_factor = 1.15 ### Feedrate adjustment factor
alt_amount = -0.4 ### Extrusion adjustment amount
alt_temp = 0 ### Temperate adjustment amount

################  Print Loop  ##################
while True:
	try:
		line = gfile_print.readline()

		if not line or linecount==final_line:
			
			print("\nFinished printing. \n")  ### End of g-code file ### 
			set_temperature(0, 0, remote_connection,0)
			time.sleep(1)
			remote_connection.send("sendgcode G0 F5500 X0 Y150\n")
			remote_connection.send("sendgcode G92 E0\n")
			time.sleep(1)
			remote_connection.send("sendgcode G28 Z\n")			
			remote_connection.send("sendgcode G28 Z\n")
			cv2.destroyAllWindows()
			command = input("\nConfirm piece is removed from print bed by hitting 'enter'. \n")
			ssh.close()
			os.remove(times_file)
			break

		else: #### Print g-code ####
			time.sleep(0.01)
			print("Layer: "+str(layercount)+", Line "+str(linecount)+" of "+str(layerbreak))

			### Change feedrate by factor of fr_factor
			if layercount > 3:
				line = adjust_feedrate_amount(line,fr_factor)

			### Change extrusion by amount alt_amount
			if layercount > 6:
				line = adjust_feedrate_amount(line,alt_amount)
				
			### Change temperature by amount alt_temp
			if layercount == 7:
				extruder_temp = extruder_temp + alt_temp
				set_temperature(extruder_temp, bed_temp, remote_connection,2)

			### Send gcode to UMS3 printer
			remote_connection.send("sendgcode "+line+"\n")

			if linecount == 33: #### Zero Bed Offset ####
				zero_bed(gfile,z_offset,remote_connection)
				print_initial_lines(remote_connection)
				for t in range(int(8)):
					print("*",end="",flush=True)			
					time.sleep(1)
				print("\n\n")
				starttime = time.time()

			if linecount == layerbreak-1:	### Get final X and Y positions of layer	
				X_line,Y_line = get_XY(line)
				if X_line > 0 and Y_line > 0:
					X = X_line
					Y = Y_line

			if linecount == layerbreak:
				print("\n\nLayer: "+str(linecount)+" , Breakpoints"+str(layercount)+"\n\n")

				if layercount % 2 == 1:
					### Retract extruder
					set_temperature(extruder_temp-150, bed_temp, remote_connection,2)
					adjust_extruder(remote_connection, 20) ## amount to extrude in mm
					### Position for camera capture
					goal_Z = Z+0.1
					goal_Y = 135+(linecount%2)/10
					out = remote_connection.recv(9999)
					remote_connection.send("sendgcode G0 F7000 X0 Y150\n")
					remote_connection.send("sendgcode G0 Z"+str(goal_Z)+"\n")
					
					### Sleep for estimated time for layer

					ts = time.strftime("%Y%m%d%H%M")
					im_3DWorkspace = 'database/3DPrinterWorkspace_'+gfile_name+'_'+str(layercount)+'_'+str(ts)+'.jpg'
					im_Top = 'database/Topview_'+gfile_name+'_'+str(layercount)+'_'+str(ts)+'.jpg'
					stl_3Dworkspace(img_size_x, img_size_y, stl_file, layercount, im_3DWorkspace, im_Top)

					endtime = time.time()
					timediff = (endtime-starttime)
					time.sleep(int(abs(timepause-timediff)))
					i = 0
					for t in range(10000000):
						if t % 2 == 1:
							print("*",end="",flush=True)
						i = check_position(0,150,goal_Z,remote_connection, layerbreak)
						if i == 1:
							print("|",end="",flush=True)
							break
						time.sleep(.85)
						
					remote_connection.send("sendgcode G0 F5000 X"+str(goal_X)+\
						" Y"+str(goal_Y)+" Z"+str(goal_Z)+"\n")
					
					i = 0		
					for t in range(10000000):
						if t % 2 == 1:
							print("-",end="",flush=True)
						i = check_position(goal_X,goal_Y,goal_Z,remote_connection, layerbreak)
						if i == 1:
							video_capture(img_size_x, img_size_y, gfile_name, gcode_file, webcam, layercount)
							break
						time.sleep(2)
						
					### Position to resume printing
					set_temperature(extruder_temp, bed_temp, remote_connection,2)
					gpositionXY = "X"+str(X)+" Y"+str(Y)
					gpositionZ = "Z"+str(Z)
					### Return to last XYZ position
					remote_connection.send("sendgcode G0 "+gpositionZ+"\n")	
					remote_connection.send("sendgcode G0 "+gpositionXY+"\n")	
					time.sleep(2)
					timepause = 0
					starttime = time.time()

				### Update layerbreak and time_break
				elapsed_time, layerbreak, Z = get_time_elapsed(times)
				timepause += elapsed_time	
				layercount += 1
				
			linecount += 1


	### Escape/Close program manually using ctrl+c ####
	except(KeyboardInterrupt):
		print("\nExit program.\n")
		remote_connection.send("sendgcode G28 Z\n")
		set_temperature(0, 0, remote_connection,0)
		time.sleep(1)
		remote_connection.send("sendgcode G0 F5500 X0 Y150\n")
		time.sleep(1)		
		remote_connection.send("sendgcode G28 Z\n")
		cv2.destroyAllWindows()
		break


