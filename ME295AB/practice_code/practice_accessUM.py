import paramiko

ip_address = "10.1.10.203"
username = "ultimaker"
password = "ultimaker"

ssh = paramiko.SSHClient()

def run_command_on_device(ip_address, username, password, command):
	""" Connect to a device, run a command, and return the output."""

	# Load SSH host keys.
	ssh.load_system_host_keys()
	# Add SSH host key when missing.
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
	total_attempts = 3
	for attempt in range(total_attempts):
		try:
			print("Attempt to connect: %s" % attempt)
			# Connect to router using username/password authentication.
			ssh.connect(ip_address, 
				username=username, 
				password=password,
				look_for_keys=False, port=22)
			# Invoke the shell of the client machine
			remote_connection = ssh.invoke_shell() 
			# Send command.
			remote_connection.send(command)
			# Print output.
			output = remote_connection.recv(10240)
			print(output)
			# Close connection.
			ssh.close()
			return output

		except Exception as error_message:
			print("Unable to connect")
			print(error_message)

# Run function

command = "sendgcode G28"
command1 = command+"\n"
router_output = run_command_on_device(ip_address, username, password, command1)


