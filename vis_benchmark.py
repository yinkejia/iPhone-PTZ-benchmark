import os
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from plyfile import PlyData, PlyElement

def load_depth_float16(path):
    depth_u16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        return None
    return depth_u16.view(np.float16).astype(np.float32)

def get_camera_ply_data(c2w, pos_color=(0, 255, 0), look_color=(0, 0, 255)):
    """
    Creates vertex data for a single camera.
    """
    pos = c2w[:3, 3]
    look_dir = c2w[:3, 2] * 0.2 
    look_at = pos + look_dir
    
    vertices = np.array([
        (pos[0], pos[1], pos[2], *pos_color),
        (look_at[0], look_at[1], look_at[2], *look_color)
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    return vertices

def reconstruct_dense_pcd(depth, K, c2w, img_path):
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
    
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().cpu().numpy()
    
    z = depth.flatten()
    mask = z > 0
    
    x = (u.flatten()[mask] - cx) * z[mask] / fx
    y = (v.flatten()[mask] - cy) * z[mask] / fy
    pts_cam = np.stack([x, y, z[mask]], axis=-1)
    pts_world = (pts_cam @ c2w[:3, :3].T) + c2w[:3, 3]
    
    img = np.array(Image.open(img_path).convert("RGB").resize((W, H)))
    colors = img.reshape(-1, 3)[mask]
    
    vertex_data = np.empty(pts_world.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'], vertex_data['y'], vertex_data['z'] = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]
    vertex_data['red'], vertex_data['green'], vertex_data['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    
    return vertex_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="iPhone-PTZ")
    parser.add_argument("--scene", type=str, default="container")
    parser.add_argument("--output_folder", type=str, default="./visualizations")
    parser.add_argument("--input_root", type=str, default="./evaluation_benchmarks")
    args = parser.parse_args()

    scene_path = os.path.join(args.input_root, args.benchmark, args.scene)
    vis_out = os.path.join(args.output_folder, args.benchmark, args.scene)
    os.makedirs(vis_out, exist_ok=True)

    src_Ks = torch.load(os.path.join(scene_path, "camera_poses/source_Ks.pt"), weights_only=True)
    src_c2ws = torch.load(os.path.join(scene_path, "camera_poses/source_c2ws.pt"), weights_only=True)
    tgt_c2ws = torch.load(os.path.join(scene_path, "camera_poses/target_c2ws.pt"), weights_only=True)

    for i in range(0, len(src_c2ws), 20):
        frame_name = f"frame_{i:05d}"
        print(f"Generating visualization for {frame_name}...")
        
        depth_path = os.path.join(scene_path, "depth_maps", f"{frame_name}_depth.png")
        img_path = os.path.join(scene_path, "source_imgs", f"{frame_name}.png")
        depth = load_depth_float16(depth_path)
        if depth is None: continue

        # 1. Reconstruct Point Cloud for current frame
        pcd_verts = reconstruct_dense_pcd(depth, src_Ks[i], src_c2ws[i], img_path)
        
        # 2. Collect ALL cameras at interval 10
        cam_vertices_list = []
        edges_list = []
        current_v_idx = len(pcd_verts)

        for j in range(0, len(src_c2ws), 10):
            # Source Cameras (Green -> Blue)
            src_v = get_camera_ply_data(src_c2ws[j].numpy(), (0, 255, 0), (0, 0, 255))
            cam_vertices_list.append(src_v)
            edges_list.append((current_v_idx, current_v_idx + 1))
            current_v_idx += 2

            # Target Cameras (Red -> Yellow)
            tgt_v = get_camera_ply_data(tgt_c2ws[j].numpy(), (255, 0, 0), (255, 255, 0))
            cam_vertices_list.append(tgt_v)
            edges_list.append((current_v_idx, current_v_idx + 1))
            current_v_idx += 2

        # Combine PCD and all camera vertices
        all_verts = np.concatenate([pcd_verts] + cam_vertices_list)
        
        # Format Edge Data
        edge_data = np.array(edges_list, dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])

        # Create PLY Elements
        v_el = PlyElement.describe(all_verts, 'vertex')
        e_el = PlyElement.describe(edge_data, 'edge')

        PlyData([v_el, e_el]).write(os.path.join(vis_out, f"{frame_name}_vis.ply"))

    print(f"Visualization files saved to {vis_out}")