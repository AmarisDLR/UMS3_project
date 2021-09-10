import cv2
import time
import calendar
import re
import paramiko

# First, find the initial temperature
# M109 and M104 do not work in SSH griffin
def find_init_temperature(gfile):
	with open(gfile) as gcode:
		for line in gcode:
			#print(line)
			line = line.strip()
			layertemp = re.findall(";EXTRUDER_TRAIN.0.INITIAL_TEMPERATURE:", line)
			if layertemp:
				temp = line.split(":",1)[1]
				return temp
				break

def set_temperature(temp, remote_connection):
	remote_connection.send("select printer printer/head/0/slot/0 \n")	
	remote_connection.send("set pre_tune_target_temperature "+ str(temp)+" \n")

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
z_offset = input("\nEnter Z-offset: \n")
z_offset = 20.001-z_offset
home_z = 236-z_offset
remote_connection.send("sendgcode G92 Z"+home_z+" \n")
out = remote_connection.recv(9999)
print(out)
################  Get & Send Temperature  ##################

print("                 \n\n\n\n")

gfile = "UMgcode/square.gcode" #input("Name of gcode file to print: <file.gcode> \n")
print(gfile+"\n")
temp = find_init_temperature(gfile)
print("Initial Temperature: " + str(temp) + " F")
set_temperature(temp, remote_connection)


################  Print g-code  ##################

print("\n\nStart printing from file.\n\n")
time.sleep(20)


with open(gfile) as gcode:
	for i, line in enumerate(gcode):
		#print(line)
		time.sleep(1)

		remote_connection.send("sendgcode " +line +" \n")
		out = remote_connection.recv(9999)
		print(out)
		if i == 50:
			break
			


################  End of Program  ##################
while True:
	try:
		print("\nFinished printing. \n")
		remote_connection.send("sendgcode G28 \n")

		command = input("\nConfirm piece is removed from print bed by hitting 'enter': \n")
		remote_connection.send("set pre_tune_target_temperature "+ "0 "+" \n")
		remote_connection.send("get current temperature \n")
		remote_connection.send("sendgcode G28 \n")
		ssh.close()
		break

	except(KeyboardInterrupt):
		print("\nExit program.\n")
		remote_connection.send("set pre_tune_target_temperature "+ "0 "+" \n")
		remote_connection.send("get current temperature \n")
		remote_connection.send("sendgcode G28 \n")
		ssh.close()

