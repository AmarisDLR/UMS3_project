import paramiko

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

commands = ["sendgcode G28","sendgcode G0 X32", "sendgcode G0 Y32", "sendgcode G0 X52", "sendgcode G0 X52"]

for command in commands:

		print(command)
		remote_connection.send(command+"\n")
		out = remote_connection.recv(9999)
		print(out)

while True:
	try:
		command = input("Enter command: ")
		#print(command)
		remote_connection.send(command)
		remote_connection.send("\n")
		output = remote_connection.recv(9999)
		print(output)

	except(KeyboardInterrupt):
		print("Exit program")
		ssh.close()


ssh.close()
