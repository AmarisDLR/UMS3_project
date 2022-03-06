"""
Send G-code to UMS3 via SSH communication.
"""

import paramiko
import re

def sendgcode(printer, gcode):
	printer.send("sendgcode "+gcode+"\n")

################  Log in to SSH  ##################
ip_address = "192.168.0.226"
username = "ultimaker"
password = "ultimaker"
ssh = paramiko.SSHClient()
### Load SSH host keys.
ssh.load_system_host_keys()
### Add SSH host key when missing.
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip_address,username=username,password=password,look_for_keys=False,port=22)
remote_connection = ssh.invoke_shell()
### send out command to begin shell program
remote_connection.send("\n")
out = remote_connection.recv(9999)
print(out)
sendgcode(remote_connection,"G0 F2500")
out = remote_connection.recv(9999)
print(out)

################  Get Times Btwn Layers  ##################

while True:
    try:
        command = input("Enter G-code: ")
        sendgcode(remote_connection,command)
        out = remote_connection.recv(9999)
        print(out)
        if re.findall("An existing connection was forcibly closed by the remote host", str(out)):
            print("\nForced exit.")
            break
    except(KeyboardInterrupt):
        print('\nExit program.')
        remote_connection.close()
        break
