import copy

import numpy as np
import torch
import math
from rdkit import Chem
import networkx as nx


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1., 1., -1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    return R, t


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def modify_conformer(coords, values, edge_index, mask_rotate):
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
    # torsion_updates = (torch.sigmoid(values[6:]) - 0.5) * np.pi * 2
    # rot_update = torch.sin(values[3:6]) * np.pi
    # torsion_updates = torch.sin(values[6:]) * np.pi
    torsion_updates = values[6:]

    lig_center = torch.mean(coords, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    # rot_mat = euler_angles_to_matrix(rot_update.squeeze(), 'XYZ')
    rigid_new_pos = (coords - lig_center) @ rot_mat.T + tr_update + lig_center

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos.clone(),
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
            return aligned_flexible_pos
        except:
            return flexible_new_pos
    else:
        return rigid_new_pos


def modify_conformer2(coords, values, edge_index, mask_rotate):
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
    # torsion_updates = (torch.sigmoid(values[6:]) - 0.5) * np.pi * 2
    # rot_update = values[3:6]
    torsion_updates = values[6:]

    lig_center = torch.mean(coords, dim=0, keepdim=True)
    rigid_new_pos = coords - lig_center

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            holo_pos = flexible_new_pos @ R.T + t.T
        except:
            holo_pos = flexible_new_pos
    else:
        holo_pos = rigid_new_pos
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    out_pos = holo_pos @ rot_mat.T + tr_update + lig_center
    return out_pos


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates):
    for idx_edge, e in enumerate(edge_index):
        # if torsion_updates[idx_edge] == 0:
        #     continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        # assert not mask_rotate[idx_edge, u]
        # assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec / torch.norm(rot_vec)  # idx_edge!
        rot_mat = gen_matrix_from_rot_vec(rot_vec, torsion_updates[idx_edge])

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    return pos


def gen_matrix_from_rot_vec(k, theta):
    K = torch.zeros((3, 3), device=k.device)
    K[[1, 2, 0], [2, 0, 1]] = -k
    K[[2, 0, 1], [1, 2, 0]] = k
    R = torch.eye(3, device=k.device) + K * torch.sin(theta) + (1 - torch.cos(theta)) * torch.matmul(K, K)
    return R


def get_mask_rotate(mol, device='cpu'):
    mol = Chem.RemoveHs(mol)
    G = nx.Graph()
    nodes = range(len(mol.GetAtoms()))
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    assert nx.is_connected(G), "分子图必须为连通图"
    torsions = []
    masks = []
    torsion_smarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsion_smarts)
    matches = mol.GetSubstructMatches(torsion_query)
    for edge in matches:
        G2 = G.to_undirected()
        G2.remove_edge(*edge)
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                torsions.append(edge)
                mask = torch.zeros(len(nodes), dtype=torch.bool, device=device)
                mask[l] = True
                masks.append(mask)
    return torsions, masks
