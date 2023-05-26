import paramiko
import socket
import threading
import sys
import logging

logging.basicConfig()
logging.getLogger("paramiko").setLevel(logging.DEBUG) # for example

class Server(paramiko.ServerInterface):
	def check_auth_password(self, username, password):
		if (username == "testuser") and (password == "testpass"):
			return paramiko.AUTH_SUCCESSFUL
		return paramiko.AUTH_FAILED

	def get_allowed_auths(self, username):
		return 'password,publickey'
	
	def check_auth_publickey(self, username, key):
		if username != "testuser":
			return paramiko.AUTH_FAILED
		
		referenceKey = paramiko.RSAKey.from_private_key_file("test-key", "testpass")
		
		if key.fingerprint == referenceKey.fingerprint:
			return paramiko.AUTH_SUCCESSFUL
		
		return paramiko.AUTH_FAILED
	
	def __init__(self):
		self.event = threading.Event()

def listener():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	sock.bind(('127.0.0.1', 2222))
	sock.settimeout(0.2)

	sock.listen(100)
	
	while True:
		try:
			client, addr = sock.accept()
		except socket.timeout:
			pass
		else:
			break

	t = paramiko.Transport(
		client
	)
		
	#t.default_window_size = 2147483647
	#t.packetizer.REKEY_BYTES = pow(2, 40)
	#t.packetizer.REKEY_PACKETS = pow(2, 40)
	
	t.set_gss_host(socket.getfqdn(""))
	print("Server moduli load status:", t.load_server_moduli("./moduli"))
	t.add_server_key(host_key)
	server = Server()
	t.start_server(server=server)

	# Wait 30 seconds for a command
	server.event.wait(1)
	t.close()

host_key = paramiko.rsakey.RSAKey.from_private_key_file("mock-ssh-key")


while True:
	try:
		listener()
	except KeyboardInterrupt:
		sys.exit(0)