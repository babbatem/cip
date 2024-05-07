import torch
import numpy as np
from ikflow import config


def geodesic_distance_between_quaternions(q1: np.array, q2: np.array) -> np.array:
    """Given rows of quaternions q1 and q2, compute the geodesic distance between each

    Args:
        q1 (np.array): _description_
        q2 (np.array): _description_

    Returns:
        np.array: _description_
    """

    assert len(q1.shape) == 2
    assert len(q2.shape) == 2
    assert q1.shape[0] == q2.shape[0]
    assert q1.shape[1] == q2.shape[1]

    q1_R9 = rotation_matrix_from_quaternion(torch.Tensor(q1).to(config.device))
    q2_R9 = rotation_matrix_from_quaternion(torch.Tensor(q2).to(config.device))
    return geodesic_distance(q1_R9, q2_R9).cpu().data.numpy()


def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """TODO: document

    Args:
        v (torch.Tensor): [batch x n]

    Returns:
        torch.Tensor: [batch x n]
    """
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(config.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def rotation_matrix_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """TODO: document

    Args:
        quaternion (torch.Tensor): [batch x 4]

    Returns:
        torch.Tensor: [batch x 3 x 3]
    """
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


def geodesic_distance(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Calculate the geodesic distance between rotation matrices

    Args:
        m1 (torch.Tensor): [batch x 3 x 3] rotation matrix
        m2 (torch.Tensor): [batch x 3 x 3] rotation matrix

    Returns:
        torch.Tensor: [batch] rotational differences between m1, m2. Between 0 and pi for each element
    """
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(config.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(config.device)) * -1)
    theta = torch.acos(cos)
    return theta
