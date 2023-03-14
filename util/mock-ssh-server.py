import paramiko
import socket
import threading
import sys

class Server(paramiko.ServerInterface):
	def check_auth_password(self, username, password):
		if (username == "testuser") and (password == "testpass"):
			return paramiko.AUTH_SUCCESSFUL
		return paramiko.AUTH_FAILED

	def get_allowed_auths(self, username):
		return 'password'
	
	def __init__(self):
		self.event = threading.Event()

def listener():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	sock.bind(('', 2222))
	sock.settimeout(0.2)

	sock.listen(100)
	
	while True:
		try:
			client, addr = sock.accept()
		except socket.timeout:
			pass
		else:
			break

	t = paramiko.Transport(client)
	t.set_gss_host(socket.getfqdn(""))
	t.load_server_moduli()
	server = Server()
	t.start_server(server=server)

	# Wait 30 seconds for a command
	server.event.wait(30)
	t.close()


while True:
	try:
		listener()
	except KeyboardInterrupt:
		sys.exit(0)