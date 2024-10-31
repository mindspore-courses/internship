import cv2
import os


if __name__=="__main__":
   
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    depth_1=cv2.imread(f"{code_dir}/demo_data/mustard0/depth/1581120424100262102.png")
    depth_2=cv2.imread(f"{code_dir}/demo_data/realDex/depth/0.png")
    print(depth_1.max(), depth_2.max())

