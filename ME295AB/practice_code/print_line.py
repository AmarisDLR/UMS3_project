import os
import time
import calendar
import re
import paramiko
import cv2

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

################ send out command to begin shell program


def set_temperature(extruder_temp, bed_temp, remote_connection, flag):
	print("\nTarget Extruder Temperature: " + str(extruder_temp) + " F\n")
	print("\nTarget Bed Temperature: " + str(extruder_temp) + " F\n")				
	remote_connection.send("select printer printer/head/0/slot/0 \n")
	remote_connection.send("set pre_tune_target_temperature "+ str(extruder_temp)+" \n")	
	time.sleep(2)
	remote_connection.send("select printer printer/bed \n")
	remote_connection.send("set pre_tune_target_temperature "+ str(bed_temp)+" \n")
	
	if flag == 1:	
		print("\n\nStart heating\n\n")
		input("\n\nStop heating\n\n")

set_temperature(202, 65, remote_connection, 1)

#remote_connection.send("sendgcode G28\n")
#out = remote_connection.recv(9999)
#print(out)
remote_connection.send("sendgcode G92 E0\n")
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G0 X25 Y175\n")
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G0 F2000 Z20\n")
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G0 F650 Z4.5\n")
out = remote_connection.recv(9999)
print(out)
time.sleep(5)
remote_connection.send("sendgcode G0 F1000 Y50 E1\n") 
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G0 X30\n") 
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G0 F1000 Y175 E2.5\n") 
out = remote_connection.recv(9999)
print(out)
remote_connection.send("sendgcode G0 Z20 \n")
out = remote_connection.recv(9999)
print(out)

time.sleep(30)
#set_temperature(0, 0, remote_connection, 0)

