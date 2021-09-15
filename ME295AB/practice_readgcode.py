import re
import time
import cv2
import calendar

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

def capture_img(camera):
	cam_fps = camera.get(cv2.CAP_PROP_FPS)
	print('Capture Image at %.2f FPS.' %cam_fps)
	ts = calendar.timegm(time.gmtime())
	imfile = str(ts)+'img.jpg'
	print(imfile)	
	cv2.imwrite(filename=imfile, img=frame)
	print("Image saved!")

def video_capture(webcam):
	count = 1
	t = True
	while t:
		try:
			count += 1
			print("\n\n"+str(count)+"\n\n")
			check, frame = webcam.read()
			#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if count == 50: 
				capture_img(webcam)
				t = False
				break
		except(KeyboardInterrupt):
			cv2.destroyAllWindows()
			break
		
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)#640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)#480)
webcam.set(cv2.CAP_PROP_FPS, 25)#30)

times_file = "times.txt"
linecount = 1
times = open(times_file,"r")

gfile_print = open(gfile, "r")

elapsed_time, layerbreak = get_time_elapsed(times)
print(elapsed_time, layerbreak)


while True:
	try:
		time.sleep(.001)
		line = gfile_print.readline()
		print(linecount)
		check, frame = webcam.read()
		#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		cv2.imshow("Capturing", frame)
		key = cv2.waitKey(1)


		if linecount == layerbreak:
			u = input("\nopen video capure\n")
			video_capture(webcam)
			elapsed_time, layerbreak = get_time_elapsed(times)
			print(elapsed_time, layerbreak)
			u = input("\nlayerbreak\n")
			

		if not line:
			print("\n\nEnd of file\n\n")
			break

		linecount += 1

	except(KeyboardInterrupt):
		webcam.release()
		cv2.destroyAllWindows()
		print("\nExit program.\n")
		break
