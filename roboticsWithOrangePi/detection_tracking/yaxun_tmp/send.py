import pickle
import socket
import struct
import time
import cv2
import numpy as np
import sys
import os
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(code_dir)

import pyk4a
from pyk4a import Config, PyK4A

k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            camera_fps=pyk4a.FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
k4a.start()
calibration = k4a.calibration
 
K = calibration.get_camera_matrix(1) # stand for color type
    # # getters and setters directly get and set on device
    # k4a.whitebalance = 4500
    # assert k4a.whitebalance == 4500
    # k4a.whitebalance = 4510
    # assert k4a.whitebalance == 4510


    # for pre_process parameters
window_name = 'k4a'
start_trigger = False
annotation = False
first_tracking_frame = False
index = 0
zfar = 2.0
first_downscale = True
shorter_side = 720

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server address
#server_address = ('10.19.125.237', 1234)
server_address = ('127.0.0.1', 1234)

s.connect(server_address)
# #max_packet_size = 65507
# max_packet_size = 65000
send_rate = 1.0 / 30.0
#send_rate = 1.0

# 发送数据
while True:
    capture = k4a.get_capture()
    if first_downscale:
            H, W = capture.color.shape[:2]
            downscale = shorter_side / min(H, W)
            H = int(H*downscale)
            W = int(W*downscale)
            K[:2] *= downscale
            first_downscale = False        
    color = capture.color[...,:3].astype(np.uint8)
    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST) 
    depth = capture.transformed_depth.astype(np.float32) / 1e3
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.01) | (depth>=zfar)] = 0

    rgbd_data = {
          'rgb' : color ,
          'depth' : depth
    }
    data = pickle.dumps(rgbd_data)
    data_size = len(data)
    print(data_size)
    data_size = struct.pack('>I', data_size)
    
    # # 发送数据大小
    # s.sendto(data_size, ('10.19.125.237', 1234))
    # # 发送实际数据
    # chunks = [data[i:i + max_packet_size] for i in range(0, len(data), max_packet_size)]
    # print(f'chunks : {len(chunks)}')
    # for chunk in chunks:
        # s.sendto(chunk, ('10.19.125.237', 1234))

    # Send data length first
    s.sendall(len(data).to_bytes(4, byteorder='big'))
        # Send the actual data
    s.sendall(data)
    # 等待下一个发送周期
    time.sleep(send_rate)

k4a.stop()
sock.close()