from typing import Tuple
import torch

POSE_VEC_TRANSLATION_INDICES = slice(0, 3)
POSE_VEC_ROTATION_INDICES = slice(3, 6)

def project_points(inverse_depth, pose, intrinsics):
    b, _, h, w = inverse_depth.shape
    intrinsics_inv = torch.inverse(intrinsics.cpu()).to(intrinsics.device)
    cam_coords = inverse_depth_to_camera_coords(inverse_depth, intrinsics_inv)
    cam_coords = cam_coords.permute(0, 2, 3, 1).reshape(-1, 3, 1)
    pose = pose.permute(0, 2, 3, 1).reshape(-1, 6)
    pose_mat = pose_vec2mat(pose)
    pose_mat = pose_mat.view(-1,  4, 4)
    transformed_points = transform_points(cam_coords, pose_mat)
    transformed_points = transformed_points.view(b, h, w, 3).permute(0, 3, 1, 2)
    transformed_points = transformed_points.view(*transformed_points.shape[:2], -1)
    pixel_coords_3d = intrinsics.bmm(transformed_points)  # shape [B, 3, ...]
    depth = pixel_coords_3d[:, 2:3]
    pixel_coords = (pixel_coords_3d[:, :2] / depth).view(b, 2, h, w)
    return pixel_coords, depth

def transform_points(camera_coords, tranformation):
    ones = torch.ones(
        [camera_coords.shape[0], 1, 1],
        dtype=camera_coords.dtype,
        device=camera_coords.device,
    )
    camera_coords = torch.cat([camera_coords, ones], dim=1)
    transformed_points = tranformation[:, :3].bmm(camera_coords)

    return transformed_points


def inverse_depth_to_camera_coords(inverse_depth: torch.Tensor, intrinsics_inv: torch.Tensor) -> torch.Tensor:
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        inverse_depth: depth map B,1,H,W
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B,3,3]
    Returns:
        array of (X,Y,Z) cam coordinates -- [B,3,H,W]
    """
    b, _, h, w = inverse_depth.shape

    # compose homogeneous tensors
    i_range = torch.arange(h).view(1, h, 1).expand(1, h, w).type_as(inverse_depth)  # [1, H, W]
    j_range = torch.arange(w).view(1, 1, w).expand(1, h, w).type_as(inverse_depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(inverse_depth)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    # expand to batch
    pixel_coords = pixel_coords.expand(b, 3, h, w)  # [B, 3, H, W]
    pixel_coords_flat =  pixel_coords.view(*pixel_coords.shape[:2], -1)
    camera_coords = intrinsics_inv.bmm(pixel_coords_flat).view_as(pixel_coords)

    # scale by depth
    # assert inverse_depth.min().item() >= 10e-5  # avoid division by zero
    return camera_coords / inverse_depth


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat

def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    translation = vec[..., POSE_VEC_TRANSLATION_INDICES].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., POSE_VEC_ROTATION_INDICES]  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def inverse_warp(source_image, target_inverse_depth, pose, intrinsics, source_inverse_depth = None, padding_mode: str = "border"):
    src_pixel_coords, computed_depth = project_points(target_inverse_depth, pose, intrinsics)
    normalisation_factors = (
        torch.tensor(
            [target_inverse_depth.shape[-1], target_inverse_depth.shape[-2]], dtype=torch.float32, device=src_pixel_coords.device
        ).view(1, 1, 1, 2)
        - 1
    )
    grid_sample_locations = 2 * src_pixel_coords.permute(0, 2, 3, 1) / normalisation_factors - 1

    projected_img = torch.nn.functional.grid_sample(
        source_image, grid_sample_locations, padding_mode=padding_mode, align_corners=False
    )
    projected_depth = None
    if source_inverse_depth is not None:
        projected_depth = torch.nn.functional.grid_sample(
            1 / source_inverse_depth, grid_sample_locations, padding_mode=padding_mode, align_corners=False
        )
    valid_mask = (src_pixel_coords.abs().max(dim=-1)[0] < 1).unsqueeze(1)

    return projected_img, computed_depth, projected_depth, valid_mask

