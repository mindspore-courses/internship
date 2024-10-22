import open3d as o3d
import numpy as np
import cv2
import json
inisvis = True




class VisualizePaM:
    def __init__(self, color, depth, K, mesh_path,pose ):
        self.K = K
        self.mesh_path = mesh_path
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=3.0,  # truncate depth at 3 meters
        convert_rgb_to_intensity=False
            )
        H, W = color.shape[:2]
        self.H = H
        self.W = W
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic( W, H, K)
            )
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.transform(pose) 
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord('s')-ord('a') + 65, self.save_camera_parameters)
        self.vis.create_window(width=W, height=H)
        self.vis.get_render_option().background_color = [255, 255, 255]
        self.vis.add_geometry(pcd)
        self.vis.add_geometry(mesh)
 
    def save_camera_parameters(self, vis):
       vis.run()
       param = vis.get_view_control().convert_to_pinhole_camera_parameters()
       o3d.io.write_pinhole_camera_parameters("camera_parameters.json", param)
       vis.create_window(width=self.W, height=self.H)


    def visualize_pcd(self, color, depth ,pose):
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
            o3d.camera.PinholeCameraIntrinsic( W, H, self.K)
            )
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        mesh.transform(pose) 


        # ctr.set_lookat([0, 0, 0])   # Point the camera is looking at
        # ctr.set_up([0, 1, 0])       # 'Up' direction in the world coordinate
        # ctr.set_front([0, 0, -1])   # Direction from the camera position to the lookat point
        # ctr.set_zoom(0.5)
        camera_params = o3d.io.read_pinhole_camera_parameters("camera_parameters.json")

        self.vis.get_view_control().convert_from_pinhole_camera_parameters(parameter=camera_params, allow_arbitrary= True)
        print(self.vis.get_view_status())
        self.vis.poll_events()

        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)
        self.vis.add_geometry(mesh)
        #print("updata pcd")
        self.vis.update_renderer()
        # vis.run()
        #vis.capture_screen_image("temp.jpg")

    def tem_visualize_pcd(self, color, depth, K,mesh_path,pose):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=3.0,  # truncate depth at 3 meters
        convert_rgb_to_intensity=False
            )

        H  = self.H
        W = self.W

        vis = o3d.visualization.Visualizer()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.get_render_option().background_color = [255, 255, 255]

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic( W, H, K)
            )
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.transform(pose) 

        vis.add_geometry(pcd)
        vis.add_geometry(mesh)

        vis.poll_events()
        vis.update_renderer()
