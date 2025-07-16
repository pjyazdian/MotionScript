import copy
import math
from scipy.interpolate import interp1d # for downsampling
import torch

################################################################################
## READ/WRITE TO FILES
################################################################################

import json
from scipy.spatial.transform import Rotation as R
from torch.cuda import device


def save_to_file(data, filename_res):
    with open(filename_res, "w") as f:
        f.write(json.dumps(data))
        print("Saved file:", filename_res)

def load_from_file(filename_res):
    with open(filename_res, "r") as f:
        data = json.load(f)
        data = {k:float(v) for k, v in data.items()} # parse values
        print("Load results from", filename_res)
    return data


################################################################################
## MISCELLANEOUS CALCULATION
################################################################################

from scipy import linalg

def mean_list(data):
    return sum(data)/len(data)

def mean_std_list(data):
    m = mean_list(data)
    s = sum((x-m)**2 for x in data)/len(data)
    return [m, s**0.5]

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


################################################################################
## LOGGING
################################################################################

import datetime
import time
from collections import defaultdict, deque


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window
    or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


################################################################################
## ANGLE TRANSFORMATION FONCTIONS
################################################################################

import roma

def rotvec_to_eulerangles(x):
    x_rotmat = roma.rotvec_to_rotmat(x)
    thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
    thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
    thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
    return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
    N = thetax.numel()
    # rotx
    rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    rotx[:,1,1] = torch.cos(thetax)
    rotx[:,2,2] = torch.cos(thetax)
    rotx[:,1,2] = -torch.sin(thetax)
    rotx[:,2,1] = torch.sin(thetax)
    roty[:,0,0] = torch.cos(thetay)
    roty[:,2,2] = torch.cos(thetay)
    roty[:,0,2] = torch.sin(thetay)
    roty[:,2,0] = -torch.sin(thetay)
    rotz[:,0,0] = torch.cos(thetaz)
    rotz[:,1,1] = torch.cos(thetaz)
    rotz[:,0,1] = -torch.sin(thetaz)
    rotz[:,1,0] = torch.sin(thetaz)
    rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
    return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
    rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
    return roma.rotmat_to_rotvec(rotmat)


################################################################################
## LOAD POSE DATA
################################################################################

import os
import numpy as np

import text2pose.config as config


def read_posescript_json(relative_filepath):
    filepath = os.path.join(config.POSESCRIPT_LOCATION, relative_filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def get_pose_data_from_file(pose_info):
    """
    Load pose data and normalize the orientation.

    Args:
        pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]

    Returns:
        pose data, torch tensor of size (1, n_joints*3), all joints considered.
    """

    # load pose data
    assert pose_info[0] in config.supported_datasets, f"Expected data from on of the following datasets: {','.join(config.supported_datasets)} (provided dataset: {pose_info[0]})."
    dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
    # axis angle representation of selected body joints
    pose = dp['poses'][pose_info[2],:].reshape(-1,3) # (n_joints, 3)
    pose = torch.as_tensor(pose).to(dtype=torch.float32)
    # normalize the global orient
    thetax, thetay, thetaz = rotvec_to_eulerangles( pose[:1,:] )
    zeros = torch.zeros_like(thetax)
    pose[0:1,:] = eulerangles_to_rotvec(thetax, thetay, zeros)
    return pose.reshape(1, -1)


rotX = lambda theta: torch.tensor(
            [[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])
rotY = lambda theta: torch.tensor(
            [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]])
rotZ = lambda theta: torch.tensor(
            [[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])

def transf(rotMat, theta_deg, values):
    theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
    return rotMat(theta_rad).mm(values.t()).t()
def get_pose_sequence_data_from_file(pose_info, normalizer_frame=None):
    """
    Load pose data and normalize the orientation.

    Args:
        pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]

    Returns:
        pose data, torch tensor of size (1, n_joints*3), all joints considered.
    """

    # load pose data
    assert pose_info[0] in config.supported_datasets, f"Expected data from on of the following datasets: {','.join(config.supported_datasets)} (provided dataset: {pose_info[0]})."
    dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
    # axis angle representation of selected body joints
    pose = dp['poses']
    trans = dp['trans']
    pose_shape = pose.shape
    pose = pose.reshape(pose_shape[0], -1, 3)  # (n_frames, n_joints, 3)

    pose = torch.as_tensor(pose).to(dtype=torch.float32)


    # normalize the global orient
    eular_orientation = np.zeros_like(trans)

    # Since we just want to get a normalized set of frames based on the first frame
    if normalizer_frame:
        thetax_first, thetay_first, thetaz_first = rotvec_to_eulerangles(pose[normalizer_frame, :1, :])
        trans_oriented_first = torch.tensor(trans[normalizer_frame]).unsqueeze(0).float()
        # trans_oriented_first = transf(rotX, 0, trans_oriented_first)
        trans_oriented_first = transf(rotX, thetax_first, trans_oriented_first)
        trans_oriented_first = transf(rotY, thetay_first, trans_oriented_first)
        trans_oriented_first = transf(rotZ, thetaz_first, trans_oriented_first)

        trans_oriented_first[0, 1] = 0 # to keep Y axis values
    for frame in range(pose_shape[0]):

        # 1. Update orientation of the current frame w.r.t. the first frame
        thetax, thetay, thetaz = rotvec_to_eulerangles( pose[frame, :1,:] )
        if normalizer_frame:
            thetaz = thetaz - thetaz_first
        else:
            thetaz = torch.zeros_like(thetax)

        pose[frame, 0:1, :] = eulerangles_to_rotvec(thetax, thetay, thetaz)

        # 2. Normalize translation:
        trans_oriented = torch.tensor(trans[frame]).unsqueeze(0).float()
        trans_oriented = transf(rotX, thetax, trans_oriented)
        trans_oriented = transf(rotY, thetay, trans_oriented)
        trans_oriented = transf(rotZ, thetaz, trans_oriented)

        if normalizer_frame:
            trans[frame] = trans_oriented-trans_oriented_first
        else:
            trans[frame] = trans_oriented

        eular_orientation[frame, 0] = thetax
        eular_orientation[frame, 1] = thetay
        eular_orientation[frame, 2] = thetaz

    # todo: translation shoud be normalized such that the starting frame be at origin.
    print("todo: translation shoud be normalized such that the starting frame be at origin.")
    return pose.reshape(pose_shape[0], -1), trans

def get_pose_sequence_data_from_file_HumanML3D(npyfile_address, normalizer_frame=None):
    """
    Load pose data and normalize the orientation.

    Args:
        pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]

    Returns:
        pose data, torch tensor of size (1, n_joints*3), all joints considered.
    """

    # load pose data
    # assert pose_info[0] in config.supported_datasets, f"Expected data from on of the following datasets: {','.join(config.supported_datasets)} (provided dataset: {pose_info[0]})."
    dp = np.load(npyfile_address)
    # axis angle representation of selected body joints

    # pose = dp['poses']
    # trans = dp['trans']
    # pose_shape = pose.shape
    # pose = pose.reshape(pose_shape[0], -1, 3)  # (n_frames, n_joints, 3)

    pose = dp[:, :, :]
    trans = dp[:,:1, :].squeeze()
    pose_shape = pose.shape
    pose = torch.as_tensor(pose).to(dtype=torch.float32)


    # normalize the global orient
    eular_orientation = np.zeros_like(trans)

    # Since we just want to get a normalized set of frames based on the first frame
    if normalizer_frame:
        thetax_first, thetay_first, thetaz_first = rotvec_to_eulerangles(pose[normalizer_frame, :1, :])
        trans_oriented_first = torch.tensor(trans[normalizer_frame]).unsqueeze(0).float()
        trans_oriented_first = transf(rotX, -90, trans_oriented_first)
        trans_oriented_first[0, 1] = 0 # to keep Y axis values
    for frame in range(pose_shape[0]):

        thetax, thetay, thetaz = rotvec_to_eulerangles( pose[frame, :1,:] )
        if normalizer_frame:
            thetaz = thetaz - thetaz_first
        else:
            thetaz = torch.zeros_like(thetax)
        # Normalize translation:
        trans_oriented = torch.tensor(trans[frame]).unsqueeze(0).float()
        trans_oriented = transf(rotX, -90, trans_oriented)
        trans[frame] = trans_oriented-trans_oriented_first

        pose[frame, 0:1,:] = eulerangles_to_rotvec(thetax, thetay, thetaz)
        eular_orientation[frame, 0] = thetax
        eular_orientation[frame, 1] = thetay
        eular_orientation[frame, 2] = thetaz

    # todo: translation shoud be normalized such that the starting frame be at origin.
    print("todo: translation shoud be normalized such that the starting frame be at origin.")
    return pose.reshape(pose_shape[0], -1), trans


def get_pose_sequence_data_from_file_HumanML3D_T2M_GPT(npyfile_address, npy_file_euler_address, normalizer_frame=None):
    def qrot(q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]

        original_shape = list(v.shape)
        # print(q.shape)
        q = q.contiguous().view(-1, 4)
        v = v.contiguous().view(-1, 3)

        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

    def qinv(q):
        assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
        mask = torch.ones_like(q)
        mask[..., 1:] = -mask[..., 1:]
        return q * mask

    def qefix(q):
        """
        Enforce quaternion continuity across the time dimension by selecting
        the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
        between two consecutive frames.

        Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
        Returns a tensor of the same shape.
        """
        assert len(q.shape) == 3
        # assert q.shape[-1] == 4

        result = q.copy()
        dot_products = np.sum(q[1:] * q[:-1], axis=2)
        mask = dot_products < 0
        mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
        result[1:][mask] *= -1
        return result

    def qeuler(q, order, epsilon=0, deg=True):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4

        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise

        if deg:
            return torch.stack((x, y, z), dim=1).view(original_shape) * 180 / np.pi
        else:
            return torch.stack((x, y, z), dim=1).view(original_shape)
    def recover_root_rot_pos(data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]

        euler_angles_deg = qeuler(r_rot_quat, "xyz", deg=False)
        euler_angles_deg = torch.Tensor(qefix(euler_angles_deg[:, None, :].cpu().numpy()).squeeze(1)).to(data.device)

        return r_rot_quat, r_pos, euler_angles_deg


        # Editted by Ahmet, keep it here for now
        # rot_vel = data[..., 0]
        # r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        # '''Get Y-axis rotation from rotation velocity'''
        # r_rot_ang[..., 1:] = rot_vel[..., :-1]
        # r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
        #
        # lin_vel = data[..., 2]
        # l_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        # '''Get Y-axis rotation from rotation velocity'''
        # l_rot_ang[..., 1:] = lin_vel[..., :-1]
        # l_rot_ang = torch.cumsum(l_rot_ang, dim=-1)
        #
        # r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        # r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        # r_rot_quat[..., 1] = torch.sin(l_rot_ang)
        # r_rot_quat[..., 2] = torch.sin(r_rot_ang)
        # r_rot_quat[..., 3] = torch.cos(l_rot_ang)
        #
        # euler_angles_deg = torch.zeros((r_rot_quat.shape[0], 3))
        # euler_angles_deg[..., 0] = torch.acos(l_rot_ang)
        # euler_angles_deg[..., 1] = torch.asin(r_rot_ang)
        # euler_angles_deg[..., 1] = torch.asin(l_rot_ang)
        #
        # r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        # r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        # '''Add Y-axis rotation to root position'''
        # r_pos = qrot(qinv(r_rot_quat), r_pos)
        #
        # r_pos = torch.cumsum(r_pos, dim=-2)
        #
        # r_pos[..., 1] = data[..., 3]
        #
        # from scipy.spatial.transform import Rotation
        #
        # # rot = Rotation.from_quat(r_rot_quat)
        # # euler_angles_deg = rot.as_euler('xyz', degrees=True)
        # # euler_angles_deg = qeuler_ABDOLLAH(r_rot_quat, order='xyz', epsilon= 0, deg=True, follow_order=True)
        #
        # # euler_angles_deg = torch.zeros((r_rot_quat.shape[0], 3))
        # # r_rot_quat_np = r_rot_quat.cpu().numpy()
        # # for ix in range(r_rot_quat.shape[0]):
        # #     # euler_angles_deg[ix] = torch.tensor(np.array(qu2eu_ABDOLGHADER(r_rot_quat_np[ix])))/np.pi * 180
        # #     euler_angles_deg[ix] = torch.tensor(np.array(euler_from_quaternion(*r_rot_quat_np[ix]))) / np.pi * 180
        #
        # # euler_angles_deg = qeuler(r_rot_quat, "xyz", deg=False)
        # euler_angles_deg = torch.Tensor(qefix(euler_angles_deg[:, None, :].cpu().numpy()).squeeze(1)).to(data.device)
        #
        # return r_rot_quat, r_pos, euler_angles_deg

    def recover_from_ric(data, joints_num):
        r_rot_quat, r_pos, euler_angles = recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        FirstPerson_positions = copy.deepcopy(positions)
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        FirstPerson_positions[..., 0] += r_pos[..., 0:1]
        FirstPerson_positions[..., 2] += r_pos[..., 2:3]


        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
        FirstPerson_positions = torch.cat([r_pos.unsqueeze(-2), FirstPerson_positions], dim=-2)

        return FirstPerson_positions, positions, r_rot_quat, r_pos, euler_angles

    def _index_from_letter(letter: str):
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2

    def _angle_from_tan(
            axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ):
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.
        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.
        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def quaternion_to_matrix(quaternions):
        """
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
    def matrix_to_euler_angles(matrix, convention: str):
        """
        Convert rotations given as rotation matrices to Euler angles in radians.
        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.
        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
        i0 = _index_from_letter(convention[0])
        i2 = _index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            _angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            _angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)

    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def quaternion_to_euler_angle_vectorized1(input):
        res = np.zeros([len(input), 3])
        for ix in range(len(input)):
            w, x, y, z = input[ix]
            ysqr = y * y

            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + ysqr)
            X = np.degrees(np.arctan2(t0, t1))

            t2 = +2.0 * (w * y - z * x)
            t2 = np.where(t2 > +1.0, +1.0, t2)
            # t2 = +1.0 if t2 > +1.0 else t2

            t2 = np.where(t2 < -1.0, -1.0, t2)
            # t2 = -1.0 if t2 < -1.0 else t2
            Y = np.degrees(np.arcsin(t2))

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (ysqr + z * z)
            Z = np.degrees(np.arctan2(t3, t4))
            res[ix] = [X, Y, Z]
        return res

        #     ------------------------------------------------------------------------------------------------------------------

    def quaternion_to_euler2x(q):
        w, x, y, z = q
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))  # use 90 degrees if out of range
        else:
            pitch = math.degrees(math.asin(sinp))

        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        return roll, pitch, yaw

    def qeuler_ABDOLLAH(q, order, epsilon=0, deg=True, follow_order=True):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4

        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise
        resdict = {"x": x, "y": y, "z": z}

        # print(order)
        reslist = [resdict[order[i]] for i in range(len(order))] if follow_order else [x, y, z]
        # print(reslist)
        if deg:
            return torch.stack(reslist, dim=1).view(original_shape) * 180 / np.pi
        else:
            return torch.stack(reslist, dim=1).view(original_shape)

    def qu2eu_ABDOLGHADER(q):
        p = -1
        q03 = q[0] ** 2 + q[3] ** 2
        q12 = q[1] ** 2 + q[2] ** 2
        xhi = np.sqrt(q03 * q12)
        if q12 == 0:
            theta = [np.arctan2(-2 * p * q[0] * q[3], q[0] ** 2 - q[3] ** 2), 0, 0]
        elif q03 == 0:
            theta = [np.arctan2(2 * q[1] * q[2], q[1] ** 2 - q[2] ** 2), np.pi, 0]
        else:
            theta = [np.arctan2((q[1] * q[3] - p * q[0] * q[2]) / xhi, (-p * q[0] * q[1] - q[2] * q[3]) / xhi),
                     np.arctan2(2 * xhi, q03 - q12),
                     np.arctan2((p * q[0] * q[2] + q[1] * q[3]) / xhi, (q[2] * q[3] - p * q[0] * q[1]) / xhi)]
        return theta

    def quaternion_to_euler_ABDOLGPT(quaternion):
        w, x, y, z = quaternion
        ysqr = y * y

        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = np.arctan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # Clamp at 1 to avoid arccos error
        pitch = np.arcsin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = np.arctan2(t3, t4)

        return [roll, pitch, yaw]  # or return in degrees: np.degrees([roll, pitch, yaw])


    motions = np.load(npyfile_address)
    # motion_tensor = torch.tensor(motions).view(1, motions.shape[0], -1)
    # mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    # std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
    motion_tensor = torch.tensor(motions).view(motions.shape[0], -1)

    FirstPerson_pred_xyz, pred_xyz, r_rot_quat, r_pos, euler_angles_from_quat = recover_from_ric((motion_tensor).float(), 22)
    # pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
    # xyz = pred_xyz.reshape(1, -1, 22, 3)
    try:
        euler_angles_from_file = np.load(npy_file_euler_address)
        euler_angles = euler_angles_from_file.squeeze()
        # euler_angles = euler_angles_from_quat
    except: # for the HumanAct12 subdataset in HumanML3D which doesn't have Euler info.
        euler_angles = np.zeros((pred_xyz.shape[0],3))
        euler_angles[:,0] = np.pi/2



    xyz = pred_xyz.reshape(pred_xyz.shape[0], -1)
    FirstPerson_xyz = FirstPerson_pred_xyz.reshape(FirstPerson_pred_xyz.shape[0], -1)
    return FirstPerson_xyz, xyz, euler_angles, r_pos



def get_pose_sequence_data_from_file_BEAT2T(npyfile_address, npy_file_euler_address, normalizer_frame=None):
    def qrot(q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]

        original_shape = list(v.shape)
        # print(q.shape)
        q = q.contiguous().view(-1, 4)
        v = v.contiguous().view(-1, 3)

        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

    def qinv(q):
        assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
        mask = torch.ones_like(q)
        mask[..., 1:] = -mask[..., 1:]
        return q * mask

    def qefix(q):
        """
        Enforce quaternion continuity across the time dimension by selecting
        the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
        between two consecutive frames.

        Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
        Returns a tensor of the same shape.
        """
        assert len(q.shape) == 3
        # assert q.shape[-1] == 4

        result = q.copy()
        dot_products = np.sum(q[1:] * q[:-1], axis=2)
        mask = dot_products < 0
        mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
        result[1:][mask] *= -1
        return result

    def qeuler(q, order, epsilon=0, deg=True):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4

        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise

        if deg:
            return torch.stack((x, y, z), dim=1).view(original_shape) * 180 / np.pi
        else:
            return torch.stack((x, y, z), dim=1).view(original_shape)
    def recover_root_rot_pos(data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]

        euler_angles_deg = qeuler(r_rot_quat, "xyz", deg=False)
        euler_angles_deg = torch.Tensor(qefix(euler_angles_deg[:, None, :].cpu().numpy()).squeeze(1)).to(data.device)

        return r_rot_quat, r_pos, euler_angles_deg


        # Editted by Ahmet, keep it here for now
        # rot_vel = data[..., 0]
        # r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        # '''Get Y-axis rotation from rotation velocity'''
        # r_rot_ang[..., 1:] = rot_vel[..., :-1]
        # r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
        #
        # lin_vel = data[..., 2]
        # l_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        # '''Get Y-axis rotation from rotation velocity'''
        # l_rot_ang[..., 1:] = lin_vel[..., :-1]
        # l_rot_ang = torch.cumsum(l_rot_ang, dim=-1)
        #
        # r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        # r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        # r_rot_quat[..., 1] = torch.sin(l_rot_ang)
        # r_rot_quat[..., 2] = torch.sin(r_rot_ang)
        # r_rot_quat[..., 3] = torch.cos(l_rot_ang)
        #
        # euler_angles_deg = torch.zeros((r_rot_quat.shape[0], 3))
        # euler_angles_deg[..., 0] = torch.acos(l_rot_ang)
        # euler_angles_deg[..., 1] = torch.asin(r_rot_ang)
        # euler_angles_deg[..., 1] = torch.asin(l_rot_ang)
        #
        # r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        # r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        # '''Add Y-axis rotation to root position'''
        # r_pos = qrot(qinv(r_rot_quat), r_pos)
        #
        # r_pos = torch.cumsum(r_pos, dim=-2)
        #
        # r_pos[..., 1] = data[..., 3]
        #
        # from scipy.spatial.transform import Rotation
        #
        # # rot = Rotation.from_quat(r_rot_quat)
        # # euler_angles_deg = rot.as_euler('xyz', degrees=True)
        # # euler_angles_deg = qeuler_ABDOLLAH(r_rot_quat, order='xyz', epsilon= 0, deg=True, follow_order=True)
        #
        # # euler_angles_deg = torch.zeros((r_rot_quat.shape[0], 3))
        # # r_rot_quat_np = r_rot_quat.cpu().numpy()
        # # for ix in range(r_rot_quat.shape[0]):
        # #     # euler_angles_deg[ix] = torch.tensor(np.array(qu2eu_ABDOLGHADER(r_rot_quat_np[ix])))/np.pi * 180
        # #     euler_angles_deg[ix] = torch.tensor(np.array(euler_from_quaternion(*r_rot_quat_np[ix]))) / np.pi * 180
        #
        # # euler_angles_deg = qeuler(r_rot_quat, "xyz", deg=False)
        # euler_angles_deg = torch.Tensor(qefix(euler_angles_deg[:, None, :].cpu().numpy()).squeeze(1)).to(data.device)
        #
        # return r_rot_quat, r_pos, euler_angles_deg

    def recover_from_ric(data, joints_num):
        r_rot_quat, r_pos, euler_angles = recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        FirstPerson_positions = copy.deepcopy(positions)
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        FirstPerson_positions[..., 0] += r_pos[..., 0:1]
        FirstPerson_positions[..., 2] += r_pos[..., 2:3]


        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
        FirstPerson_positions = torch.cat([r_pos.unsqueeze(-2), FirstPerson_positions], dim=-2)

        return FirstPerson_positions, positions, r_rot_quat, r_pos, euler_angles

    def _index_from_letter(letter: str):
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2

    def _angle_from_tan(
            axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ):
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.
        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.
        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def quaternion_to_matrix(quaternions):
        """
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
    def matrix_to_euler_angles(matrix, convention: str):
        """
        Convert rotations given as rotation matrices to Euler angles in radians.
        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.
        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
        i0 = _index_from_letter(convention[0])
        i2 = _index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            _angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            _angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)

    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def quaternion_to_euler_angle_vectorized1(input):
        res = np.zeros([len(input), 3])
        for ix in range(len(input)):
            w, x, y, z = input[ix]
            ysqr = y * y

            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + ysqr)
            X = np.degrees(np.arctan2(t0, t1))

            t2 = +2.0 * (w * y - z * x)
            t2 = np.where(t2 > +1.0, +1.0, t2)
            # t2 = +1.0 if t2 > +1.0 else t2

            t2 = np.where(t2 < -1.0, -1.0, t2)
            # t2 = -1.0 if t2 < -1.0 else t2
            Y = np.degrees(np.arcsin(t2))

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (ysqr + z * z)
            Z = np.degrees(np.arctan2(t3, t4))
            res[ix] = [X, Y, Z]
        return res

        #     ------------------------------------------------------------------------------------------------------------------

    def quaternion_to_euler2x(q):
        w, x, y, z = q
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))  # use 90 degrees if out of range
        else:
            pitch = math.degrees(math.asin(sinp))

        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        return roll, pitch, yaw

    def qeuler_ABDOLLAH(q, order, epsilon=0, deg=True, follow_order=True):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4

        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise
        resdict = {"x": x, "y": y, "z": z}

        # print(order)
        reslist = [resdict[order[i]] for i in range(len(order))] if follow_order else [x, y, z]
        # print(reslist)
        if deg:
            return torch.stack(reslist, dim=1).view(original_shape) * 180 / np.pi
        else:
            return torch.stack(reslist, dim=1).view(original_shape)

    def qu2eu_ABDOLGHADER(q):
        p = -1
        q03 = q[0] ** 2 + q[3] ** 2
        q12 = q[1] ** 2 + q[2] ** 2
        xhi = np.sqrt(q03 * q12)
        if q12 == 0:
            theta = [np.arctan2(-2 * p * q[0] * q[3], q[0] ** 2 - q[3] ** 2), 0, 0]
        elif q03 == 0:
            theta = [np.arctan2(2 * q[1] * q[2], q[1] ** 2 - q[2] ** 2), np.pi, 0]
        else:
            theta = [np.arctan2((q[1] * q[3] - p * q[0] * q[2]) / xhi, (-p * q[0] * q[1] - q[2] * q[3]) / xhi),
                     np.arctan2(2 * xhi, q03 - q12),
                     np.arctan2((p * q[0] * q[2] + q[1] * q[3]) / xhi, (q[2] * q[3] - p * q[0] * q[1]) / xhi)]
        return theta

    def quaternion_to_euler_ABDOLGPT(quaternion):
        w, x, y, z = quaternion
        ysqr = y * y

        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = np.arctan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # Clamp at 1 to avoid arccos error
        pitch = np.arcsin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = np.arctan2(t3, t4)

        return [roll, pitch, yaw]  # or return in degrees: np.degrees([roll, pitch, yaw])
    loaded = dict(np.load(npyfile_address))
    # We need to resample to 20-FPS since the motionscript was originally
    # developed with that and other FPS affects thresholds, velocity, timing, etc.
    pose_seq = loaded['poses']
    pose_seq_20fps = interp1d(np.linspace(0, len(pose_seq) - 1, len(pose_seq)), pose_seq, axis=0)(
        np.linspace(0, len(pose_seq) - 1, int(len(pose_seq) * 20 / 30)))
    # Todo: do the same for the translation

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pose_seq_data =  torch.from_numpy(loaded['poses']).to(device)

    import text2pose.utils_visu as utils_visu
    from human_body_prior.body_model.body_model import BodyModel
    body_model = BodyModel(bm_fname=config.SMPLH_NEUTRAL_BM, num_betas=config.n_betas)
    body_model.eval()
    body_model.to(device)
    # infer coordinates
    with torch.no_grad():
        j_seq_3D_coord = body_model(pose_body=pose_seq_data[:, 3:66],
                                    pose_hand=pose_seq_data[:, 66+3*3:],
                                    root_orient=pose_seq_data[:, :3]).Jtr  # U should decide about the root orentation that I want to use
    j_seq = j_seq_3D_coord.detach().cpu()

    # BEAT data has 3 more joints after 21 body and before 30 hands
    pose_seq_data = torch.cat((pose_seq_data[:, :3],         # root orientation
                              pose_seq_data[:, 3:66],       # body angle axis
                              pose_seq_data[:, 66+3*3:]), dim=1)    # hands angle axis

    # Not required:
    # for frame_i in range(j_seq.shape[0]):
    #     j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])
    j_seq = pose_seq_data.detach().cpu().numpy()

    motions = j_seq
    # motion_batchified = np.expand_dims(motions, axis=0)
    # title, name = ["Test"], 'ABBAS'
    # motion_batchified = torch.unsqueeze(motions, 0)
    # utils_visu.draw_to_batch_Payam(motion_batchified, title, [f'out_temp/{name}_3DS.gif'])


    # motions = loaded['poses']
    # motions = motions[:, :21*3+3].reshape((-1, 22, 3))
    global_orientation = loaded['poses'][:, :3].reshape((-1, 3))

    motion_tensor = torch.tensor(motions)
    global_orientation = torch.from_numpy(global_orientation)
    euler_angles = global_orientation

    # motion_tensor = torch.tensor(motions).view(1, motions.shape[0], -1)
    # mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    # std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
    # motion_tensor = torch.tensor(motions).view(motions.shape[0], -1)

    # FirstPerson_pred_xyz, pred_xyz, r_rot_quat, r_pos, euler_angles_from_quat = recover_from_ric((motion_tensor).float(), 22)
    FirstPerson_pred_xyz, pred_xyz = motion_tensor, motion_tensor
    r_rot_quat, r_pos, euler_angles_from_quat = None, torch.tensor(loaded['trans']), None

    # motion_tensor_translated = torch.zeros_like(motion_tensor)
    # for fr in range(motion_tensor.shape[0]):
    #     motion_tensor_translated[fr] = motion_tensor[fr, :] + r_pos[fr]
    # pred_xyz = motion_tensor_translated


    # pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
    # xyz = pred_xyz.reshape(1, -1, 22, 3)

    # try:
    #     euler_angles_from_file = np.load(npy_file_euler_address)
    #     euler_angles = euler_angles_from_file.squeeze()
    #     # euler_angles = euler_angles_from_quat
    # except: # for the HumanAct12 subdataset in HumanML3D which doesn't have Euler info.
    #     euler_angles = np.zeros((pred_xyz.shape[0],3))
    #     euler_angles[:,0] = np.pi/2



    xyz = pred_xyz.reshape(pred_xyz.shape[0], -1)
    FirstPerson_xyz = FirstPerson_pred_xyz.reshape(FirstPerson_pred_xyz.shape[0], -1)
    return FirstPerson_xyz, xyz, euler_angles, r_pos

def get_pose_sequence_data_from_file_Salsa_Dance(npyfile_address, npy_file_euler_address, normalizer_frame=None):
    # Shay
    loaded = dict(np.load(npyfile_address, allow_pickle=True))
    # We need to resample to 20-FPS since the motionscript was originally
    # developed with that and other FPS affects thresholds, velocity, timing, etc.
    limit_frames = 200
    loaded['poses'] = loaded['poses'][:limit_frames]
    loaded['trans'] = loaded['trans'][:limit_frames]

    pose_seq = loaded['poses']
    pose_seq_20fps = interp1d(np.linspace(0, len(loaded['poses']) - 1, len(loaded['poses'])), loaded['poses'], axis=0)(
        np.linspace(0, len(loaded['poses']) - 1, int(len(loaded['poses']) * 20 / 30)))
    trans_20fps = interp1d(np.linspace(0, len(loaded['trans']) - 1, len(loaded['trans'])), loaded['trans'], axis=0)(
        np.linspace(0, len(loaded['trans']) - 1, int(len(loaded['trans']) * 20 / 30)))
    loaded['poses'] = pose_seq_20fps
    loaded['trans'] = trans_20fps

    poses_rotvec4mesh = loaded['poses'][:, :66].copy()

    # for fr_i in range(loaded['poses'].shape[0]):
    #     loaded['poses'][fr_i, :3] = (R.from_rotvec(loaded['poses'][fr_i, :3]) * R.from_euler('x', -90, degrees=True)).as_rotvec()
    #     # loaded['poses'][fr_i, :3] = (R.from_rotvec(loaded['poses'][fr_i, :3]) * R.from_euler('y', -90, degrees=True)).as_rotvec()
    # loaded['trans'] = transf(rotX, -90, torch.tensor(loaded['trans']).float()).cpu().detach().numpy()
    # Todo: do the same for the translation

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from smplx import SMPLX
    def smplx_to_pos3d_SFU(data):
        smplx = None

        # smplx = SMPLX(model_path='/localhome/cza152/Desktop/Duolando/smplx/models/smplx',
        #                betas=data['betas'][:, :10], gender=data['meta']['gender'], \
        #     batch_size=len(data['betas']), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)
        frames = data['poses'].shape[0]
        b = np.repeat(data['betas'][:10], frames).reshape((frames, 10))
        smplx = SMPLX(model_path='SMPLX_DEP\\models_lockedhead\\smplx', betas=b,
                      gender=np.array2string(data['gender'])[1:-1], \
                      batch_size=len(b), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)

        keypoints3d = smplx.forward(
            # global_orient=torch.from_numpy(data['global_orient']).float(),

            global_orient=torch.from_numpy(data['poses'][:, :3], ).float(),
            body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
            jaw_pose=torch.from_numpy(data['poses'][:, 66:69]).float(),
            leye_pose=torch.from_numpy(data['poses'][:, 69:72]).float(),
            reye_pose=torch.from_numpy(data['poses'][:, 72:75]).float(),
            left_hand_pose=torch.from_numpy(data['poses'][:, 75:120]).float(),
            right_hand_pose=torch.from_numpy(data['poses'][:, 120:]).float(),
            transl=torch.from_numpy(data['trans']).float(),  # transl=torch.from_numpy(data['transl']).float(),
            # betas=torch.from_numpy(data['betas'][:10]).float()
            betas=torch.from_numpy(b).float()
        ).joints.detach().numpy()[:, :55]

        nframes = keypoints3d.shape[0]
        return keypoints3d
    pose_seq_data = smplx_to_pos3d_SFU(loaded)
    j_seq = torch.tensor(pose_seq_data)

    # # Not required:
    for frame_i in range(j_seq.shape[0]):
        j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])
    # j_seq = pose_seq_data.detach().cpu().numpy()

    motions = j_seq
    # motion_batchified = np.expand_dims(motions, axis=0)
    # title, name = ["Test"], 'ABBAS'
    # motion_batchified = torch.unsqueeze(motions, 0)
    # utils_visu.draw_to_batch_Payam(motion_batchified, title, [f'out_temp/{name}_3DS.gif'])


    # motions = loaded['poses']
    # motions = motions[:, :21*3+3].reshape((-1, 22, 3))

    # quickly rotagte just the root orientation
    for fr_i in range(loaded['poses'].shape[0]):
        rotation_fix = R.from_euler('x', -90, degrees=True)
        loaded['poses'][fr_i, :3] = (R.from_rotvec(loaded['poses'][fr_i, :3]) * rotation_fix).as_rotvec()
    global_orientation = loaded['poses'][:, :3].reshape((-1, 3))

    motion_tensor = torch.tensor(motions)
    global_orientation = torch.from_numpy(global_orientation)
    euler_angles = global_orientation


    FirstPerson_pred_xyz, pred_xyz = motion_tensor, motion_tensor
    r_rot_quat, r_pos, euler_angles_from_quat = None, torch.tensor(loaded['trans']), None




    xyz = pred_xyz.reshape(pred_xyz.shape[0], -1)
    FirstPerson_xyz = FirstPerson_pred_xyz.reshape(FirstPerson_pred_xyz.shape[0], -1)

    return FirstPerson_xyz, xyz, euler_angles, r_pos, poses_rotvec4mesh


def get_pose_sequence_data_from_file_MOTIONX(npyfile_address, normalizer_frame=None):
    def qrot(q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]

        original_shape = list(v.shape)
        # print(q.shape)
        q = q.contiguous().view(-1, 4)
        v = v.contiguous().view(-1, 3)

        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

    def qinv(q):
        assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
        mask = torch.ones_like(q)
        mask[..., 1:] = -mask[..., 1:]
        return q * mask

    def qefix(q):
        """
        Enforce quaternion continuity across the time dimension by selecting
        the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
        between two consecutive frames.

        Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
        Returns a tensor of the same shape.
        """
        assert len(q.shape) == 3
        # assert q.shape[-1] == 4

        result = q.copy()
        dot_products = np.sum(q[1:] * q[:-1], axis=2)
        mask = dot_products < 0
        mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
        result[1:][mask] *= -1
        return result

    def qeuler(q, order, epsilon=0, deg=True):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4

        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise

        if deg:
            return torch.stack((x, y, z), dim=1).view(original_shape) * 180 / np.pi
        else:
            return torch.stack((x, y, z), dim=1).view(original_shape)
    def recover_root_rot_pos(data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]

        euler_angles_deg = qeuler(r_rot_quat, "xyz", deg=False)
        euler_angles_deg = torch.Tensor(qefix(euler_angles_deg[:, None, :].cpu().numpy()).squeeze(1)).to(data.device)

        return r_rot_quat, r_pos, euler_angles_deg


        # Editted by Ahmet, keep it here for now
        # rot_vel = data[..., 0]
        # r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        # '''Get Y-axis rotation from rotation velocity'''
        # r_rot_ang[..., 1:] = rot_vel[..., :-1]
        # r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
        #
        # lin_vel = data[..., 2]
        # l_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        # '''Get Y-axis rotation from rotation velocity'''
        # l_rot_ang[..., 1:] = lin_vel[..., :-1]
        # l_rot_ang = torch.cumsum(l_rot_ang, dim=-1)
        #
        # r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        # r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        # r_rot_quat[..., 1] = torch.sin(l_rot_ang)
        # r_rot_quat[..., 2] = torch.sin(r_rot_ang)
        # r_rot_quat[..., 3] = torch.cos(l_rot_ang)
        #
        # euler_angles_deg = torch.zeros((r_rot_quat.shape[0], 3))
        # euler_angles_deg[..., 0] = torch.acos(l_rot_ang)
        # euler_angles_deg[..., 1] = torch.asin(r_rot_ang)
        # euler_angles_deg[..., 1] = torch.asin(l_rot_ang)
        #
        # r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        # r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        # '''Add Y-axis rotation to root position'''
        # r_pos = qrot(qinv(r_rot_quat), r_pos)
        #
        # r_pos = torch.cumsum(r_pos, dim=-2)
        #
        # r_pos[..., 1] = data[..., 3]
        #
        # from scipy.spatial.transform import Rotation
        #
        # # rot = Rotation.from_quat(r_rot_quat)
        # # euler_angles_deg = rot.as_euler('xyz', degrees=True)
        # # euler_angles_deg = qeuler_ABDOLLAH(r_rot_quat, order='xyz', epsilon= 0, deg=True, follow_order=True)
        #
        # # euler_angles_deg = torch.zeros((r_rot_quat.shape[0], 3))
        # # r_rot_quat_np = r_rot_quat.cpu().numpy()
        # # for ix in range(r_rot_quat.shape[0]):
        # #     # euler_angles_deg[ix] = torch.tensor(np.array(qu2eu_ABDOLGHADER(r_rot_quat_np[ix])))/np.pi * 180
        # #     euler_angles_deg[ix] = torch.tensor(np.array(euler_from_quaternion(*r_rot_quat_np[ix]))) / np.pi * 180
        #
        # # euler_angles_deg = qeuler(r_rot_quat, "xyz", deg=False)
        # euler_angles_deg = torch.Tensor(qefix(euler_angles_deg[:, None, :].cpu().numpy()).squeeze(1)).to(data.device)
        #
        # return r_rot_quat, r_pos, euler_angles_deg

    def recover_from_ric(data, joints_num):
        r_rot_quat, r_pos, euler_angles = recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        FirstPerson_positions = copy.deepcopy(positions)
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        FirstPerson_positions[..., 0] += r_pos[..., 0:1]
        FirstPerson_positions[..., 2] += r_pos[..., 2:3]


        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
        FirstPerson_positions = torch.cat([r_pos.unsqueeze(-2), FirstPerson_positions], dim=-2)

        return FirstPerson_positions, positions, r_rot_quat, r_pos, euler_angles

    def _index_from_letter(letter: str):
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2

    def _angle_from_tan(
            axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ):
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.
        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.
        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def quaternion_to_matrix(quaternions):
        """
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
    def matrix_to_euler_angles(matrix, convention: str):
        """
        Convert rotations given as rotation matrices to Euler angles in radians.
        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.
        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
        i0 = _index_from_letter(convention[0])
        i2 = _index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            _angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            _angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)

    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def quaternion_to_euler_angle_vectorized1(input):
        res = np.zeros([len(input), 3])
        for ix in range(len(input)):
            w, x, y, z = input[ix]
            ysqr = y * y

            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + ysqr)
            X = np.degrees(np.arctan2(t0, t1))

            t2 = +2.0 * (w * y - z * x)
            t2 = np.where(t2 > +1.0, +1.0, t2)
            # t2 = +1.0 if t2 > +1.0 else t2

            t2 = np.where(t2 < -1.0, -1.0, t2)
            # t2 = -1.0 if t2 < -1.0 else t2
            Y = np.degrees(np.arcsin(t2))

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (ysqr + z * z)
            Z = np.degrees(np.arctan2(t3, t4))
            res[ix] = [X, Y, Z]
        return res

        #     ------------------------------------------------------------------------------------------------------------------

    def quaternion_to_euler2x(q):
        w, x, y, z = q
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))  # use 90 degrees if out of range
        else:
            pitch = math.degrees(math.asin(sinp))

        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        return roll, pitch, yaw

    def qeuler_ABDOLLAH(q, order, epsilon=0, deg=True, follow_order=True):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4

        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise
        resdict = {"x": x, "y": y, "z": z}

        # print(order)
        reslist = [resdict[order[i]] for i in range(len(order))] if follow_order else [x, y, z]
        # print(reslist)
        if deg:
            return torch.stack(reslist, dim=1).view(original_shape) * 180 / np.pi
        else:
            return torch.stack(reslist, dim=1).view(original_shape)

    def qu2eu_ABDOLGHADER(q):
        p = -1
        q03 = q[0] ** 2 + q[3] ** 2
        q12 = q[1] ** 2 + q[2] ** 2
        xhi = np.sqrt(q03 * q12)
        if q12 == 0:
            theta = [np.arctan2(-2 * p * q[0] * q[3], q[0] ** 2 - q[3] ** 2), 0, 0]
        elif q03 == 0:
            theta = [np.arctan2(2 * q[1] * q[2], q[1] ** 2 - q[2] ** 2), np.pi, 0]
        else:
            theta = [np.arctan2((q[1] * q[3] - p * q[0] * q[2]) / xhi, (-p * q[0] * q[1] - q[2] * q[3]) / xhi),
                     np.arctan2(2 * xhi, q03 - q12),
                     np.arctan2((p * q[0] * q[2] + q[1] * q[3]) / xhi, (q[2] * q[3] - p * q[0] * q[1]) / xhi)]
        return theta

    def quaternion_to_euler_ABDOLGPT(quaternion):
        w, x, y, z = quaternion
        ysqr = y * y

        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = np.arctan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # Clamp at 1 to avoid arccos error
        pitch = np.arcsin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = np.arctan2(t3, t4)

        return [roll, pitch, yaw]  # or return in degrees: np.degrees([roll, pitch, yaw])


    loaded = (np.load(npyfile_address))
    # We need to resample to 20-FPS since the motionscript was originally
    # developed with that and other FPS affects thresholds, velocity, timing, etc.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loaded = torch.tensor(loaded).float().to(device)
    motion_parms = {
        'root_orient': loaded[:, :3],  # controls the global root orientation
        'pose_body': loaded[:, 3:3 + 63],  # controls the body
        'pose_hand': loaded[:, 66:66 + 90],  # controls the finger articulation
        'pose_jaw': loaded[:, 66 + 90:66 + 93],  # controls the yaw pose
        'face_expr': loaded[:, 159:159 + 50],  # controls the face expression
        'face_shape': loaded[:, 209:209 + 100],  # controls the face shape
        'trans': loaded[:, 309:309 + 3],  # controls the global body position
        'betas': loaded[:, 312:],  # controls the body shape. Body shape is static
    }

    # pose_seq_20fps = interp1d(np.linspace(0, len(pose_seq) - 1, len(pose_seq)), pose_seq, axis=0)(
    #     np.linspace(0, len(pose_seq) - 1, int(len(pose_seq) * 20 / 30)))
    # Todo: do the same for the translation

    import text2pose.utils_visu as utils_visu
    from human_body_prior.body_model.body_model import BodyModel
    body_model = BodyModel(bm_fname=config.SMPLH_NEUTRAL_BM, num_betas=config.n_betas)
    body_model.eval()
    body_model.to(device)
    # infer coordinates
    # with torch.no_grad():
    #     j_seq_3D_coord = body_model(pose_body=pose_seq_data[:, 3:66],
    #                                 pose_hand=pose_seq_data[:, 66+3*3:],
    #                                 root_orient=pose_seq_data[:, :3]).Jtr  # U should decide about the root orentation that I want to use
    # j_seq = j_seq_3D_coord.detach().cpu()
    #
    # # BEAT data has 3 more joints after 21 body and before 30 hands
    # pose_seq_data = torch.cat((pose_seq_data[:, :3],         # root orientation
    #                           pose_seq_data[:, 3:66],       # body angle axis
    #                           pose_seq_data[:, 66+3*3:]), dim=1)    # hands angle axis

    pose_seq_data = torch.cat((loaded[:, :3],         # root orientation
                              loaded[:, 3:66],       # body angle axis
                              loaded[:, 66:66+90]), dim=1)    # hands angle axis

    # Not required:
    j_seq = pose_seq_data.cpu().view(pose_seq_data.shape[0], -1, 3)
    # for frame_i in range(j_seq.shape[0]):
    #     # j_seq[frame_i, 0, :] = transf(rotX, 90, j_seq[frame_i, 0, :].unsqueeze(0)).squeeze()
    #     # j_seq[frame_i, 0, :] = transf(rotY, 90, j_seq[frame_i, 0, :].unsqueeze(0)).squeeze()
    #     j_seq[frame_i, 0, :] = transf(rotZ, 90, j_seq[frame_i, 0, :].unsqueeze(0)).squeeze()
    #     j_seq[frame_i, 0, :] = transf(rotY, -90, j_seq[frame_i, 0, :].unsqueeze(0)).squeeze()
    # j_seq = pose_seq_data.detach().cpu().numpy()

    motions = (j_seq.view(j_seq.shape[0], -1)).to(device)
    # motion_batchified = np.expand_dims(motions, axis=0)
    # title, name = ["Test"], 'ABBAS'
    # motion_batchified = torch.unsqueeze(motions, 0)
    # utils_visu.draw_to_batch_Payam(motion_batchified, title, [f'out_temp/{name}_3DS.gif'])
    motion_tensor = torch.tensor(motions)

    # motions = loaded['poses']
    # motions = motions[:, :21*3+3].reshape((-1, 22, 3))


    global_orientation =  motion_parms['root_orient'].reshape((-1, 3))
    # global_orientation = torch.from_numpy(global_orientation)
    root_rotvec_angles = global_orientation

    # rad2deg = lambda theta_rad: 180.0 * theta_rad / math.pi
    # for frame in range(euler_angles.shape[0]):
    #
    #     # 1. Update orientation of the current frame w.r.t. the first frame
    #     thetax, thetay, thetaz = rotvec_to_eulerangles(global_orientation[frame, :].unsqueeze(0))
    #     euler_angles[frame, 0] = rad2deg(thetax)
    #     euler_angles[frame, 1] = rad2deg(thetay)
    #     euler_angles[frame, 2] = rad2deg(thetaz)





    # motion_tensor = torch.tensor(motions).view(1, motions.shape[0], -1)
    # mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    # std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
    # motion_tensor = torch.tensor(motions).view(motions.shape[0], -1)

    # FirstPerson_pred_xyz, pred_xyz, r_rot_quat, r_pos, euler_angles_from_quat = recover_from_ric((motion_tensor).float(), 22)
    FirstPerson_pred_xyz, pred_xyz = motion_tensor, motion_tensor
    # r_rot_quat, r_pos, euler_angles_from_quat = None, torch.tensor(loaded['trans']), None
    r_rot_quat, r_pos, euler_angles_from_quat = None, loaded[:, 309:309 + 3], None

    r_pos = motion_parms['trans']



    # motion_tensor_translated = torch.zeros_like(motion_tensor)
    # for fr in range(motion_tensor.shape[0]):
    #     motion_tensor_translated[fr] = motion_tensor[fr, :] + r_pos[fr]
    # pred_xyz = motion_tensor_translated


    # pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
    # xyz = pred_xyz.reshape(1, -1, 22, 3)

    # try:
    #     euler_angles_from_file = np.load(npy_file_euler_address)
    #     euler_angles = euler_angles_from_file.squeeze()
    #     # euler_angles = euler_angles_from_quat
    # except: # for the HumanAct12 subdataset in HumanML3D which doesn't have Euler info.
    #     euler_angles = np.zeros((pred_xyz.shape[0],3))
    #     euler_angles[:,0] = np.pi/2




    xyz = pred_xyz.reshape(pred_xyz.shape[0], -1)
    FirstPerson_xyz = FirstPerson_pred_xyz.reshape(FirstPerson_pred_xyz.shape[0], -1)



    # Convert to the location
    # j_seq = body_model(pose_body=pose_seq_data[:, 3:66],
    #                    pose_hand=pose_seq_data[:, 66:],
    #                    root_orient=pose_seq_data[:, :3]).Jtr



    return FirstPerson_xyz, xyz, root_rotvec_angles, r_pos
