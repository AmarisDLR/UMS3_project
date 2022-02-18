import paramiko
import re
import time

def command(remote_connection, line):
	matchM105 = re.findall("M105",line)
	if line != "" and line[0] != ";" and not matchM105:
		print(line)
		remote_connection.send("sendgcode "+line+"\n")
		out_line = remote_connection.recv(9999)
		while True:
			out_line = remote_connection.recv(9999)
			remote_connection.send("\n")
			print(str(out_line)+"\n")
			time.sleep(0.2)
			if out_line == b'\r\n':# or re.findall(r"b'\\r\\n\\r\\n",str(out_line)):
				break
	else:
		print(line)


ip_address = "192.168.0.226"
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

while True:
	try:
		line = input("\nEnter command: ")
		print(command)
		remote_connection.send(command)
		remote_connection.send("\n")
		output = remote_connection.recv(9999)
		print(output)


	except(KeyboardInterrupt):
		print("Exit program")
		ssh.close()
		break
