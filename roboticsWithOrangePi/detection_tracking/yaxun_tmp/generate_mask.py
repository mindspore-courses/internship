import os
import sys

from scipy.spatial.transform import Rotation
import open3d as o3d
import trimesh
import numpy as np
import cv2
import json
import imageio

# generate mask from groundtruth data
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 

sys.path.append(code_dir)

def seven_num2matrix(seven_num):
    translation = seven_num[:3]
    roatation = seven_num[3:]
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3, 3] = translation
    return transform_matrix

def create_mask(depth_image):
    # print(depth_image)
    mask = np.where(depth_image < 1, 255, 0).astype(np.uint8)

    return mask

def extrinsic(pose):

    pose = seven_num2matrix(pose)
 
    transfer = np.array( [
        [
            0.9946494438314785,
            0.07454436532416706,
            0.07152357292633452,
            1.3718472595770228
        ],
        [
            -0.020637633757839904,
            -0.53500300592373,
            0.8445980533516828,
            -0.47491973985338354
        ],
        [
            0.10122535235112222,
            -0.8415550613305062,
            -0.5306020229799359,
            1.3027908747296506
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])
    return np.linalg.inv(transfer).dot(pose)

def generate_mask_for_realDex(mesh_file="demo_data/realDex/mesh/air_duster.obj", 
                              pose=[1.356385692870416326e+00, 3.546232685546230057e-02, 9.533656933708657411e-01, 5.448309218414713051e-01, 7.885733288759733117e-01, -1.627025980516069448e-01, -2.341777875420636978e-01],):

    

    intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 912.96875 , 912.6904296875 , 963.6746826171875, 544.5726928710938)

    mesh = o3d.io.read_triangle_mesh(f"{code_dir}/{mesh_file}")
    mesh.compute_vertex_normals()


    # o3d.visualization.draw_geometries([mesh])
    renderer = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
    renderer.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())

    renderer.setup_camera(intrinsic, extrinsic(pose))


    depth_image = renderer.render_to_depth_image()
    
    depth_image = np.asarray(depth_image)
    # cv2.imshow("Mask", depth_image)
    # # Create mask from depth image
    print(depth_image.min(), depth_image.max())
    mask = create_mask(depth_image)
    # # Save or display the mask image
    cv2.imwrite(f"{code_dir}/demo_data/realDex/masks/mask.png", mask)

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


rgb = imageio.imread('/home/yaxun/FoundationPose/demo_data/realDex/00temp/0.jpg')
img_copy = rgb.copy()
box_coords = []
def click_event(event, x, y, flags, params):
    
    global img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'You clicked on ({x}, {y})')
        box_coords.append((x, y))
        
        if len(box_coords) == 2:
            # Draw a rectangle between the two points
            x0, y0 = box_coords[0]
            x1, y1 = box_coords[1]
            cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.imshow('image', img_copy)
            return
        else:
            # Draw a circle at the point
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('image', img_copy)


def generate_bounding_box_for_realDex():
    
   
    H,W = rgb.shape[:2]
 
    cv2.imshow('image',rgb)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    mask = np.zeros([H, W], dtype=np.uint8)
    mask [box_coords[0][1]:box_coords[1][1], box_coords[0][0]:box_coords[1][0]] = 255
    

    cv2.imwrite(f"{code_dir}/demo_data/realDex/masks/mask.png", mask)

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    # generate_mask_for_realDex()
    generate_bounding_box_for_realDex()

