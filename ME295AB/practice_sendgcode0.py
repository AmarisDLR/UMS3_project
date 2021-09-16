import os
import time
import calendar
import re
import paramiko
import cv2

def capture_img(camera,frame):
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
			time.sleep(0.001)
			count += 1
			#print("\n\n"+str(count)+"\n\n")
			check, frame = webcam.read()
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if count == 800: 
				capture_img(webcam, frame)
				t = False
				time.sleep(2)
				break
		except(KeyboardInterrupt):
			cv2.destroyAllWindows()
			break

def check_position(X,Y,Z,remote_connection):
	t = 1
	while t:
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
					print("\n\nMATCH\n\n")
					t == 0
					break


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
remote_connection.send("help\n")
out = remote_connection.recv(9999)
#print(out)

################  Start Camera  ##################
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)#640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)#480)
webcam.set(cv2.CAP_PROP_FPS, 25)#30)
##################################################

gfile = "UMgcode/practice.gcode"
remote_connection.send("sendgcode G28 \n")
remote_connection.send("sendgcode G0 X22 Y100 Z100\n")
out = remote_connection.recv(9999)
time.sleep(15)
#commands = ["sendgcode G28 Z200","sendgcode G0 Z32", "M400"] #, "get current_temperature", "sendgcode M114","sendgcode M114","sendgcode M114"]

#for command in commands:

		#print(command)
		#remote_connection.send(command+"\n")
		#out = remote_connection.recv(9999)
		#print(out)
gfile_print = open(gfile, "r")
linecount = 1
while True:
	try:
		line = gfile_print.readline()
		#for command in commands:
		if not line: ### End of File ###
			print("\nFinished printing. \n")
			remote_connection.send("sendgcode G28 X Y \n")
			ssh.close()
			cv2.destroyAllWindows()
			break

		else: #### Print g-code ####
			time.sleep(0.001)
			remote_connection.send("sendgcode " +line +" \n")
			#out = remote_connection.recv(9999)
			#print(out)
			if linecount%500 ==0: ## Take image
				remote_connection.send("sendgcode G0 X"+str(22+linecount/100)+ " Y"+str(100+linecount/100)+"\n")
				print("\n\nLine " + str(linecount))
				print("\n\nPause\n\n")

				out = remote_connection.recv(9999)
				check_position(22+linecount/100,100+linecount/100,100,remote_connection)

				video_capture(webcam)
				z = input("\n\ncontinue\n\n")
			linecount += 1

	except(KeyboardInterrupt):
		print("Exit program")
		ssh.close()
		cv2.destroyAllWindows()


ssh.close()
