import socket
import pickle
import cv2

import threading
lock = threading.Lock()
rgbd_data = {}
isconnect = False
def recv_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise Exception("Socket closed prematurely")
        data += more
    return data


def connect_remote_rgbd(arg):
    global isconnect
    global rgbd_data
    server_address = ('0.0.0.0', 1234)  # Adjust IP if needed, 0.0.0.0 listens on all interfaces
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(1)

    print("Waiting for a connection")
    connection, client_address = sock.accept()

    try:
        print(f"Connection from {client_address}")
        while True:
            # First, read the length of the data
            data_length_bytes = recv_all(connection, 4)
            data_length = int.from_bytes(data_length_bytes, byteorder='big')
            
            # Now read the actual data
            data = recv_all(connection, data_length)
            rgbd_data = pickle.loads(data)
            isconnect = True
            # cv2.imshow('rgb',rgbd_data['rgb'])
            # if cv2.waitKey(1) == ord('q'):  # Wait for the 'q' key to quit
            #     break
            # print(f"Received data: RGB size {rgbd_data['rgb'].shape}, Depth size {rgbd_data['depth'].shape}")

    finally:
        connection.close()
        sock.close()


def get_rgbd_data():
    global rgbd_data
    with lock:
        return rgbd_data.copy()

def getConnectState():
    global isconnect
    return isconnect
#connect_remote_rgbd(1)