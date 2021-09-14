import paramiko
import time

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
print(out)

commands = ["sendgcode G28 Z200","sendgcode G0 Z32", "M400"] #, "get current_temperature", "sendgcode M114","sendgcode M114","sendgcode M114"]

#for command in commands:

		#print(command)
		#remote_connection.send(command+"\n")
		#out = remote_connection.recv(9999)
		#print(out)

count = 0
while True:
	try:
		for command in commands:
			count += 1
			time.sleep(1)
			#command = input("Enter command: ")
			#print(command)
			remote_connection.send(command)
			remote_connection.send("\n")
			output = remote_connection.recv(9999)
			print(output)
			print(str(count)+"\n")
			print("\n")
		
		
		#break

	except(KeyboardInterrupt):
		print("Exit program")
		ssh.close()


ssh.close()
