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
    normalisation_factors = (
        torch.tensor(
            [inverse_depth.shape[-1], inverse_depth.shape[-2]], dtype=torch.float32, device=pixel_coords.device
        ).view(1, 1, 1, 2)
        - 1
    )
    pixel_coords = 2 * pixel_coords.permute(0, 2, 3, 1) / normalisation_factors - 1
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
    

    projected_img = torch.nn.functional.grid_sample(
        source_image, src_pixel_coords, padding_mode=padding_mode, align_corners=False
    )
    projected_depth = None
    if source_inverse_depth is not None:
        projected_depth = torch.nn.functional.grid_sample(
            1 / source_inverse_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False
        )
    valid_mask = (src_pixel_coords.abs().max(dim=-1)[0] < 1).unsqueeze(1)
    return projected_img, computed_depth, projected_depth, valid_mask


def compute_rigid_flow(inverse_depth, pose, intrinsics, normalised_coords=True) -> torch.Tensor:
    h, w = inverse_depth.shape[-2:]
    normalised_image_grid = _get_normalised_image_grid(h, w).to(inverse_depth.device)
    scale = torch.tensor([w / 2.0, h / 2.0], dtype=torch.float32).view(1, 1, 1, 2).to(inverse_depth.device)
    normalised_pixel_coords, _ = project_points(inverse_depth, pose, intrinsics)
    if normalised_coords:
        scale = 1
    rigid_flow = scale * (normalised_pixel_coords - normalised_image_grid)
    return rigid_flow.permute(0, 3, 1, 2)

def _get_normalised_image_grid(height: int, width: int) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x, y)
    x_grid, y_grid = torch.t(x_grid).view(1, height, width, 1), torch.t(y_grid).view(1, height, width, 1)
    return torch.cat([x_grid, y_grid], dim=-1).float()

def perception_similarity_loss(img1: torch.Tensor, img2: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    if img1.size() != img2.size():
        raise ValueError('img1 img2 sizes must match, got {} and {}.'.format(img1.size(), img2.size()))
    img1_bchw = img1.view(-1, *img1.shape[-3:])
    img2_bchw = img2.view(-1, *img2.shape[-3:])
    # minimise the distance
    sim_loss = alpha * ssim_index(img2_bchw, img1_bchw) + (1 - alpha) * torch.abs(img2_bchw - img1_bchw)
    sim_loss = sim_loss.mean(dim=-3, keepdim=True)
    return sim_loss.view(*img1.shape[:-3], 1, *img1.shape[-2:])

def ssim_index(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    The sturctural similarity index originally proposed by Z.Wang et al.:
    "Z. Wang et al: "Image quality assessment: from error visibility to structural similarity. TIP, 2004"
    The implementation is mostly based on:
    https://github.com/yzcjtr/GeoNet/blob/cf336f6b91a16997d67c5ca0c6666f0b6700c435/geonet_model.py#L249

    Args:
        x: img1 [B,3,H,W]
        y: img2 [B,3,H,W]
    Returns:
        Similarity measure between `x` and `y` [B,H,W]
    """
    c_1 = 0.01 ** 2
    c_2 = 0.03 ** 2

    mu_x = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
    mu_y = torch.nn.functional.avg_pool2d(y, 3, 1, 1)

    sigma_x = torch.nn.functional.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = torch.nn.functional.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_index_n = (2 * mu_x * mu_y + c_1) * (2 * sigma_xy + c_2)
    ssim_index_d = (mu_x ** 2 + mu_y ** 2 + c_1) * (sigma_x + sigma_y + c_2)

    ssim = ssim_index_n / ssim_index_d

    return torch.clamp((1 - ssim) / 2, 0, 1)


def spatial_gradient(tensor):
    """Compute spatial gradient, assuming H,W dimensions are the last two dimensions of tensor"""
    dy = tensor[..., 1:, :] - tensor[..., :-1, :]
    dx = tensor[..., 1:] - tensor[..., :-1]
    return dx, dy


def edge_aware_smooth_loss(x, aux=None, normalise=True, reduction='mean'):
    """
    Edge-Aware Smoothness Loss as reported in:

    @article{zhan2018unsupervised,
      title={Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction},
      author={Zhan, Huangying and Garg, Ravi and Weerasekera, Chamara Saroj and
        Li, Kejie and Agarwal, Harsh and Reid, Ian},
      journal={arXiv preprint arXiv:1803.03893},
      year={2018}
    }
    Computes smoothness loss as a function of the x map and auxiliary map e.g. image intensities.

    :param x: depth map tensor [Bx1xHxW]
    :param aux: image intensity map [Bx1xHxW]
    :param normalise: normalise x before computing smoothness
    :return:
    Loss as a scalar.
    """
    if normalise:
        mean_x = x.abs().mean((-2, -1), keepdim=True)
        x = x / (mean_x + 1e-7)

    d_dx, d_dy = spatial_gradient(x)
    aux = torch.mean(aux, dim=1)
    i_dx, i_dy = spatial_gradient(aux)
    if reduction == 'mean':
        i_dx = torch.mean(i_dx, dim=-3, keepdim=True)
        i_dy = torch.mean(i_dy, dim=-3, keepdim=True)
    elif reduction == 'max':
        i_dx, _ = torch.max(i_dx, dim=-3, keepdim=True)
        i_dy, _ = torch.max(i_dy, dim=-3, keepdim=True)
    else:
        raise ValueError("Reduction must be sum or max got {}".format(reduction))
    w_dx = torch.exp(-i_dx.abs())
    w_dy = torch.exp(-i_dy.abs())

    loss_dx = torch.mean(d_dx.abs() * w_dx)
    loss_dy = torch.mean(d_dy.abs() * w_dy)
    return loss_dx + loss_dy

def flow_inverse_warp(source_flow, target_flow):
    h, w = source_flow.shape[-2:]
    normalised_image_grid = _get_normalised_image_grid(h, w).to(target_flow.device)
    source_coords = target_flow.permute(0, 2, 3, 1) + normalised_image_grid
    reconsutructed_target_flow = torch.nn.functional.grid_sample(
        source_flow, source_coords, padding_mode='border', align_corners=False
    )
    return reconsutructed_target_flow

def compute_flow_mask(real_flow, reconstructed_flow, alpha1=0.1, alpha2=0.5):
    flow_difference = torch.norm(real_flow - reconstructed_flow, dim=1, keepdim=True)
    flow_mag = torch.norm(real_flow, dim=1, keepdim=True) + torch.norm(reconstructed_flow, dim=1, keepdim=True)
    occ_mask = flow_difference < alpha1*flow_mag + alpha2
    return occ_mask