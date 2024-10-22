import cv2
import numpy as np
import sys
import os
from objectdetection import lodegroundingDINO,detectobject,boxes2masks
from visualizPaM import VisualizePaM

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(code_dir)

import pyk4a
from pyk4a import Config, PyK4A

from estimater import *
from datareader import *
import argparse

class ClickEventHandler:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image
        self.coords = []
        cv2.setMouseCallback(self.window_name, self.click_event)
        cv2.waitKey(0)

    def click_event(self,event, x, y, flags, params):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'You clicked on ({x}, {y})')
            self.coords.append((x, y))
            
            if len(self.coords) == 2:
                print(f'already achieve 2 coordination, {self.coords}')
                # Draw a rectangle between the two points
                x0, y0 = self.coords[0]
                x1, y1 = self.coords[1]
                cv2.rectangle(self.image, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.imshow(self.window_name, self.image)
                self.active = False
            else:
                # Draw a circle at the point
                cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(self.window_name, self.image)
                cv2.waitKey(1)

    def get_coords(self):
        return self.coords


def vis_pcd(color, depth, K):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=3.0,  # truncate depth at 3 meters
            convert_rgb_to_intensity=False
            )

        H, W = color.shape[:2]
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic( W, H, K)
            )
        o3d.visualization.draw_geometries([pcd]) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/realDex/mesh/air_duster.obj')
    parser.add_argument('--mesh_file', type=str, default=f'/home/yaxun/dataset/models/kangaroo_toy.obj')
    parser.add_argument('--K_file', type=str, default=f'{code_dir}/demo_data/realDex/cam_K.txt')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/realDex')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=3)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    # parser.add_argument('--prompt',type=str,default='the bottle')
    parser.add_argument('--prompt',type=str,default='the toy')
    args = parser.parse_args()
    
    set_logging_format()
    set_seed(0)
    #
    groundingDINO_mode = lodegroundingDINO()

    # foundation pose init

    mesh = trimesh.load(args.mesh_file)

    K = np.loadtxt(args.K_file).reshape(3,3)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # k4a init
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

    while 1:
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

   
        if not annotation and np.any(color):
            cv2.imshow(window_name, color)
            key =cv2.waitKey(10)
            if key == ord('s'):
                 start_trigger = True
                 
            elif key == ord('c'):
                 cv2.destroyAllWindows()
                 exit()
        if not start_trigger:
            continue

        

        # for mask annotation
        if not annotation:
            annotation = True 

            ###############
            # event_handler = ClickEventHandler(window_name, np.array(color, dtype=np.uint8))
           
            # box_coords = event_handler.get_coords()

            # # box_coords = [(669, 442), (777, 782)]
            # mask = np.zeros([H, W], dtype=np.uint8)
            # mask [box_coords[0][1]:box_coords[1][1], box_coords[0][0]:box_coords[1][0]] = 255

            boxes=detectobject(args.prompt,color,groundingDINO_mode)
            masks=boxes2masks(boxes,W,H)
            print(f'decte mask num : {len(masks)}')
            if len(masks) != 0 :
                mask = masks[0]
            else:
                mask = np.zeros([H, W], dtype=np.uint8)
            ##############################
            # cv2.imwrite(f"{code_dir}/demo_data/realDex/masks/mask.png", mask)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)
            first_tracking_frame = True
            cv2.destroyAllWindows()
            continue
        
        
        # for foundationPose tracking
        if first_tracking_frame :
            first_tracking_frame = False
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            print(f'pose = {pose}')
            my_vis = VisualizePaM(color,depth,K,args.mesh_file,pose)

            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, K)
           
                valid = depth>=0.01
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                # o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
               
        else:
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
            my_vis.visualize_pcd(color, depth, pose)
        
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{index}.txt', pose.reshape(4,4))
    
        if debug>=1:
          center_pose = pose@np.linalg.inv(to_origin)
          vis = draw_posed_3d_box(K, img=color, ob_in_cam= center_pose, bbox=bbox)
          vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
          cv2.imshow('1', vis[...,::-1])
          key =cv2.waitKey(10)
        #   if key == ord('p'):
        #       print("pause")
        #       input("Press any key to continue...")
          
        
    
        if debug>=2:
          os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
          imageio.imwrite(f'{debug_dir}/track_vis/{index}.jpg', vis)

        index = index + 1

    k4a.stop()

