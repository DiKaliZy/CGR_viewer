import numpy as np
import math


def numpy_append(np_list, append_list, axis=0):
    if list(np.shape(np_list))[0] == 0:
        np_list = append_list
    else:
        np_list = np.append(np_list, append_list, axis=axis)
    return np_list


# R2 - R1
def angular_diff(R1, R2):
    diff = log((R1.T) @ R2)
    return diff


# rv2 - rv1
def rot_vec_diff(rv1, rv2):
    R1 = exp(rv1)
    R2 = exp(rv2)
    diff = angular_diff(R1, R2)
    return diff


def lerp(T1, T2, t):
    T = (np.array(T1) * (1.0 - t)) + (np.array(T2) * t)
    return T


def slerp(R1, R2, t):
    R = (R1.T)@R2
    if R[0, 0] > 1.0:
        R[0, 0] = 1.0
    if R[1, 1] > 1.0:
        R[1, 1] = 1.0
    if R[2, 2] > 1.0:
        R[2, 2] = 1.0
    '''if R[0, 0] == 1.0 or R[1, 1] == 1.0 or R[2, 2] == 1.0 :
        print("R", R)
        print("resul : ", R[0, 0] , R[1, 1] , R[2, 2])'''
    if np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1)/2) == 0:
        ret = R1
    elif np.dot(t*log(R),t*log(R)) == 0:
        ret = R1
    else:
        ret = R1@exp(t*log((R1.T)@R2))
    return ret


def slerp_list(R1_list, R2_list, t):
    list_len = np.shape(R1_list)[0]
    slerped_list = []
    for index in range(list_len):
        R1 = R1_list[index]
        R2 = R2_list[index]
        slerped_R = slerp(R1, R2, t)
        slerped_list.append(slerped_R)
    return np.array(slerped_list)


def log(R):
    cos_val = (R[0, 0] + R[1, 1] + R[2, 2] - 1)/2
    if cos_val > 1:
        cos_val = 1
    elif cos_val < -1:
        cos_val = -1

    angl = np.arccos(cos_val)
    if angl != 0:
        v1 = (R[2, 1]-R[1, 2])/(2*np.sin(angl))
        v2 = (R[0, 2]-R[2, 0])/(2*np.sin(angl))
        v3 = (R[1, 0]-R[0, 1])/(2*np.sin(angl))
        r = np.array([v1, v2, v3]) * angl
    else:
        v1 = (R[2, 1] - R[1, 2])
        v2 = (R[0, 2] - R[2, 0])
        v3 = (R[1, 0] - R[0, 1])
        r = np.array([v1, v2, v3])
    return r


def exp(rv):
    theta = l2norm(rv)
    axis = normalized(rv)
    R = get_rot_mat_from(axis, theta)
    return R


def l2norm(v):
    return np.sqrt(np.dot(v, v))


def normalized(v):
    l = l2norm(v)
    if l == 0.0:
        return 1 * np.array(v)
    return 1/l * np.array(v)


# euler rotation angle 받아서 rotation matrix 생성
def get_rot_mat_from(axis, theta):
    R = np.array([[np.cos(theta) + axis[0]*axis[0]*(1-np.cos(theta)),
                   axis[0]*axis[1]*(1-np.cos(theta))-axis[2]*np.sin(theta),
                   axis[0]*axis[2]*(1-np.cos(theta))+axis[1]*np.sin(theta)],
                  [axis[1]*axis[0]*(1-np.cos(theta))+axis[2]*np.sin(theta),
                   np.cos(theta)+axis[1]*axis[1]*(1-np.cos(theta)),
                   axis[1]*axis[2]*(1-np.cos(theta))-axis[0]*np.sin(theta)],
                  [axis[2]*axis[0]*(1-np.cos(theta))-axis[1]*np.sin(theta),
                   axis[2]*axis[1]*(1-np.cos(theta))+axis[0]*np.sin(theta),
                   np.cos(theta)+axis[2]*axis[2]*(1-np.cos(theta))]
                 ])
    return R


def gen_rotmat(origin, target):
    ori = (origin) / np.sqrt((origin) @ np.transpose(origin))
    targ = (target) / np.sqrt((target) @ np.transpose(target))
    rotaxis = np.cross(ori, targ)
    rotaxis = rotaxis / np.sqrt(rotaxis @ np.transpose(rotaxis))
    newz = np.cross(rotaxis, targ)
    newz = newz / np.sqrt(newz @ np.transpose(newz))
    aftrot = np.column_stack((targ, rotaxis, newz))
    befz = np.cross(rotaxis, origin)
    befz = befz / np.sqrt(befz @ np.transpose(befz))
    befrot = np.column_stack((origin, rotaxis, befz))
    if np.linalg.det(befrot) == 0:
        rotmat = np.identity(3)
    else:
        rotmat = aftrot * np.linalg.inv(befrot)
    mat = np.identity(4)
    mat[:3, :3] = rotmat

    return mat, rotmat


def cos_rule(a_len, b_len, c_len):
    a_square = a_len * a_len
    b_square = b_len * b_len
    c_square = c_len * c_len
    angle = math.acos(np.clip((a_square + b_square - c_square) / (2 * a_len * b_len), -1., 1.))
    return angle


def make_affine(transform=None, translation=None, affine_type="translation"):
    affine = np.identity(4, dtype=float)
    if affine_type == "translation":
        affine[:3, 3] = translation
    else:
        affine[:3, :3] = transform
    return affine


def dot_mat_sequence(mat_list, target_index):
    result_mat = np.identity(3)
    for index in range(target_index + 1):
        result_mat = result_mat @ mat_list[index]
    return result_mat


def get_rot_axis_from(rot_mat):
    rot_vec = log(rot_mat)
    rot_axis = normalized(rot_vec)
    rot_radians = l2norm(rot_vec)
    return rot_axis, rot_radians


def spot_light_projection_mat(light_source, root_loc):
    start = np.array(light_source[:3])
    end = np.array(root_loc)
    direction_vec = normalized(start - end)
    t = (-end[1]) / direction_vec[1]
    proj_vec_x = np.array([1, 0, 0, t * direction_vec[0]])
    proj_vec_y = np.array([0, 0, 0, 0.001])
    proj_vec_z = np.array([0, 0, 1, t * direction_vec[2]])
    projection_mat = np.row_stack((proj_vec_x, proj_vec_y, proj_vec_z))
    return projection_mat


def direction_light_projection_mat(light_source, root_loc):
    start = np.array(light_source[:3])
    end = np.array(root_loc)
    direction_vec = normalized(end-start)
    proj_vec_x = np.array([1, -direction_vec[0]/direction_vec[1], 0, 0])
    proj_vec_y = np.array([0, 0, 0, 0.001])
    proj_vec_z = np.array([0, -direction_vec[2]/direction_vec[1], 1, 0])
    projection_mat = np.row_stack((proj_vec_x, proj_vec_y, proj_vec_z))
    return projection_mat