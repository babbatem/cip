import open3d as o3d
import numpy as np
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task',   required=True, type=str, default="DoorCIP",
                    help='name of task to visualize point cloud for')

args = parser.parse_args()

pcd = o3d.io.read_point_cloud("./pointclouds/"+args.task+".ply")
world_frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

o3d.visualization.draw_geometries([pcd,world_frame_axes])
