import socket
import time
import threading
from phe import paillier
import numpy as np
import pickle

class Server:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = socket.gethostname()
        self.port = 15462
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

    def add(self, pk, sk):
        client_socket, addr = self.server_socket.accept()

        global compute

        data = client_socket.recv(1024)

        num1, num2 = pickle.loads(data)

        start_time = time.time()
        result = num1 + num2
        end_time = time.time()

        compute += end_time - start_time


        client_socket.send(pickle.dumps(result))
        client_socket.recv(1024)
        client_socket.close()

    def SM(self, pk, sk):
        client_socket, addr = self.server_socket.accept()
        global compute

        data = client_socket.recv(1024)
        num1, num2 = pickle.loads(data)

        start_time = time.time()
        num1 = sk.decrypt(num1)
        num2 = sk.decrypt(num2)
        result = num1 * num2
        result = pk.encrypt(result)
        end_time = time.time()
        #print(sk.decrypt(result))
        compute += end_time - start_time

        client_socket.send(pickle.dumps(result))
        client_socket.recv(1024)
        client_socket.close()

    def compare(self, pk, sk):
        client_socket, addr = self.server_socket.accept()
        global compute

        data = client_socket.recv(1024)

        num1, num2 = pickle.loads(data)

        start_time = time.time()
        #num1 = sk.decrypt(num1)
        #num2 = sk.decrypt(num2)
        if num1 <= num2:
            result = pk.encrypt(int(1))
        else:
            result = pk.encrypt(int(0))
        end_time = time.time()

        compute += end_time - start_time


        client_socket.send(pickle.dumps(result))
        client_socket.recv(1024)
        client_socket.close()

    def SSED(self, pk, sk):
        client_socket, addr = self.server_socket.accept()

        global compute

        data = client_socket.recv(1024)

        num = pickle.loads(data)

        start_time = time.time()
        #num1 = sk.decrypt(num1)
        #num2 = sk.decrypt(num2)
        result = pk.encrypt(num)
        end_time = time.time()
        #print(sk.decrypt(result))
        compute += end_time - start_time


        client_socket.send(pickle.dumps(result))
        client_socket.recv(1024)
        client_socket.close()


class Client:
    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = socket.gethostname()
        self.port = 15432
        self.client_socket.connect((self.host, self.port))

    def add(self, num1, num2):

        global communication


        data = pickle.dumps((num1, num2))

        start_time = time.time()
        self.client_socket.send(data)


        # time.sleep(1)

        result = self.client_socket.recv(1024)
        result = pickle.loads(result)
        #print(sk.decrypt(result))
        end_time = time.time()

        self.client_socket.send(b'ACK')

        communication += end_time - start_time

        self.client_socket.close()

    def SM(self, num1, num2):

        global communication


        data = pickle.dumps((num1, num2))

        start_time = time.time()
        self.client_socket.send(data)


        # time.sleep(1)

        result = self.client_socket.recv(1024)
        result = pickle.loads(result)
        #print(sk.decrypt(result))
        end_time = time.time()

        self.client_socket.send(b'ACK')

        communication += end_time - start_time

        self.client_socket.close()

    def compare(self, num1, num2):

        global communication


        data = pickle.dumps((num1, num2))

        start_time = time.time()
        self.client_socket.send(data)

        # time.sleep(1)

        result = self.client_socket.recv(1024)
        result = pickle.loads(result)
        #print(sk.decrypt(result))
        end_time = time.time()

        self.client_socket.send(b'ACK')

        communication += end_time - start_time

        self.client_socket.close()

    def SSED(self, num1, num2):

        global communication
        global compute

        start_time = time.time()

        data = np.linalg.norm(num1 - num2)
        data = pickle.dumps(data ** 2)
        end_time1 = time.time()
        compute += end_time1 - start_time
        self.client_socket.send(data)

        # time.sleep(1)

        result = self.client_socket.recv(1024)
        result = pickle.loads(result)
        #print(sk.decrypt(result))
        end_time = time.time()

        self.client_socket.send(b'ACK')

        communication += end_time - start_time

        self.client_socket.close()

def run_server(pk, sk, id):
    server = Server()
    if id == 1:
        server.add(pk, sk)
    elif id == 2:
        server.SM(pk, sk)
    elif id == 3:
        server.compare(pk, sk)
    elif id == 4:
        server.SSED(pk, sk)

def run_client(num1, num2, id):
    client = Client()
    if id == 1:
        client.add(num1, num2)
    elif id == 2:
        client.SM(num1, num2)
    elif id == 3:
        client.compare(num1, num2)
    elif id == 4:
        client.SSED(num1, num2)


if __name__ == "__main__":

    communication = 0
    compute = 0
    connected = 0
    kappa = 512
    pk, sk = paillier.generate_paillier_keypair(n_length=kappa)

    for i in range(1000):
        num1 = np.random.rand(256)
        num2 = np.random.rand(256)
        #print(num1)
        #print(num2)
        #num1 = pk.encrypt(float(num1))
        #num2 = pk.encrypt(float(num2))

        s1 = time.time()
        server_thread = threading.Thread(target=run_server, args=(pk, sk, 4))
        client_thread = threading.Thread(target=run_client, args=(num1, num2, 4))

        server_thread.start()
        client_thread.start()

        server_thread.join()
        client_thread.join()

        connected += time.time() - s1

    communication -= compute
    connected -= communication + compute
    print("communication:{}".format(communication))
    print("compute:{}".format(compute))
    print("connected:{}".format(connected))
