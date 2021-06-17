import numpy as np
import utility
import math
import copy

import character_model


def interpolate_posture(pose1, pose2, t):
    if t < 0 or t > 1:
        return None
    g_position1 = pose1.take_g_position_list()
    g_position2 = pose2.take_g_position_list()
    l_rotation1 = pose1.take_l_rotmat_list()
    l_rotation2 = pose2.take_l_rotmat_list()

    interpolated_position = (1-t) * g_position1 + t * g_position2
    interpolated_rotation = utility.slerp_list(l_rotation1, l_rotation2, t)
    interpolated_posture = character_model.Pose()
    interpolated_posture.set_g_position_list(interpolated_position)
    interpolated_posture.set_l_rotmat_list(interpolated_rotation)
    return interpolated_posture

#time warping
def warp_time(motion, scale_func):
    """
    :param motion:
    :param scale_func:
    :return: time warped motion, time warped motion's end frame
    """
    poses = motion.get_poses()
    max_frame = motion.get_max_frame()
    warped_motion = copy.deepcopy(motion)
    warped_poses = []
    frame = 0
    t = scale_func(frame)
    while t <= max_frame - 1:
        ceil = math.ceil(t)
        floor = math.floor(t)
        t = t - floor
        pos1 = poses[floor]
        pos2 = poses[ceil]
        interpolated_pos = interpolate_posture(pos1, pos2, t)
        warped_poses.append(interpolated_pos)
        frame = frame + 1
        t = scale_func(frame)
    warped_motion.set_max_frame(frame)
    warped_motion.set_name("warped_" + motion.get_name())
    warped_motion.set_poses(warped_poses)
    return warped_motion, frame


def get_posture_difference(pose1, pose2):
    g_position1 = pose1.take_g_position_list()
    g_position2 = pose2.take_g_position_list()
    l_rotation1 = pose1.take_l_rotmat_list()
    l_rotation2 = pose2.take_l_rotmat_list()
    position_diff = g_position2 - g_position1
    rotation_diff = []
    for index in range(np.shape(l_rotation2)[0]):
        r_diff = l_rotation1[index].T @ l_rotation2[index]
        rotation_diff.append(r_diff)
    return position_diff, np.array(rotation_diff)


def get_motion_difference(motion1, motion2):
    motion1_len = motion1.get_max_frame()
    pos_diff_list = []
    rot_diff_list = []
    poses1 = motion1.get_poses()
    poses2 = motion2.get_poses()
    for index in range(motion1_len):
        pose1 = poses1[index]
        pose2 = poses2[index]
        pos_diff, rot_diff = get_posture_difference(pose1, pose2)
        pos_diff_list.append(pos_diff)
        rot_diff_list.append(rot_diff)
    pos_diff_list = np.array(pos_diff_list)
    rot_diff_list = np.array(rot_diff_list)
    return pos_diff_list, rot_diff_list


def warp_motion(motion, target_frame, keyframe_posture, start_delta, end_delta, weight_func):
    """
    :param motion:
    :param target_frame:
    :param keyframe_posture:
    :param start_delta: the frame delta where the warping starts
                        (start frame >= 0; start frame = target_frame - start_delta)
    :param end_delta: the frame delta where the warping ends
                        (end frame <= max frame; end frame = target_frame + end_delta)
    :param weight_func: the function f about time t(time = end frame - start frame), where 0<=f(t)<=1
                        it generally return 1 when keyframe
    :return:
    """
    original_posture = motion.get_poses()[target_frame]
    temp_motion = copy.deepcopy(motion)
    keyframe_pose_diff, keyframe_rot_diff = get_posture_difference(original_posture, keyframe_posture)
    start_frame = target_frame - start_delta
    end_frame = target_frame + end_delta
    if start_frame < 0:
        start_frame = 0
    motion_len = motion.get_max_frame()
    if end_frame >= motion_len:
        end_frame = motion_len
    temp_poses = temp_motion.get_poses()
    for frame in range(start_frame, end_frame):
        t = (frame - target_frame)
        if t < 0:
            t = 1 - (t / (start_frame - target_frame))
        else:
            t = 1 - (t / (end_frame - 1 - target_frame))
        weight = weight_func(t)
        weighted_pose_diff = weight * keyframe_pose_diff

        temp_posture = temp_poses[frame]
        temp_posture.set_g_position_list(temp_posture.take_g_position_list() + weighted_pose_diff)
        rotmat_list = temp_posture.take_l_rotmat_list()
        moded_rots = []
        for index in range(np.shape(keyframe_rot_diff)[0]):
            weighted_rot_diff = utility.exp(weight * utility.log(keyframe_rot_diff[index]))
            moded_rot = rotmat_list[index] @ weighted_rot_diff
            moded_rots.append(moded_rot)
        temp_posture.set_l_rotmat_list(np.array(moded_rots))
    return temp_motion


def align_motion(motion1, motion2, motion1_align_frame=-1, motion2_align_frame=0):
    pose1 = motion1.get_poses()[motion1_align_frame]
    pose2 = motion2.get_poses()[motion2_align_frame]
    aligned_motion = copy.deepcopy(motion2)
    pos_diff, rot_diff = get_posture_difference(pose2, pose1)
    root_pos_diff = pos_diff[0]
    root_pos_diff = np.array([root_pos_diff[0], 0., root_pos_diff[2]])
    root_rot_diff = rot_diff[0]
    root_rot_diff_vec = utility.log(root_rot_diff)
    root_rot_diff_vec[0] = 0.
    root_rot_diff_vec[2] = 0.
    root_rot_diff = utility.exp(root_rot_diff_vec)
    motion_length = motion2.get_max_frame()
    aligned_poses = aligned_motion.get_poses()
    pose_zero = aligned_poses[0]
    pos_zero = pose_zero.take_g_position_list()
    for index in range(motion_length):
        pose = aligned_poses[index]
        pos = pose.take_g_position_list()
        rot = pose.take_l_rotmat_list()
        joint_num = pose.get_joint_num()
        for joint in range(joint_num):
            if joint == 0:
                rot[joint] = root_rot_diff @ rot[joint]
            pos[joint] = pos[joint] + root_pos_diff
            pos_from_origin = pos[joint] - pos_zero[0]
            rot_pos = root_rot_diff @ pos_from_origin
            pos[joint] = rot_pos + pos_zero[0]
        pose.set_g_position_list(pos)
        pose.set_l_rotmat_list(rot)
    return aligned_motion, motion_length


def concat_motion(motion1, motion2):
    poses1 = motion1.get_poses()
    poses2 = motion2.get_poses()
    concated_poses = np.concatenate((poses1, poses2), axis=0)
    concated_motion = copy.deepcopy(motion1)
    concated_motion.set_poses(concated_poses)
    new_max_frame = np.shape(poses1)[0] + np.shape(poses2)[0]
    concated_motion.set_max_frame(new_max_frame)
    concated_motion.set_name(concated_motion.get_name() + "_concated_motion")
    return concated_motion, new_max_frame


def clip_motion(motion, start_frame, end_frame, clip_name=""):
    cliped_motion = motion
    poses = cliped_motion.get_poses()
    frame_size = end_frame - start_frame
    cliped_motion.set_max_frame(frame_size)
    cliped_motion.set_name(motion.get_name() + clip_name)
    cliped_poses = poses[start_frame:end_frame]
    cliped_motion.set_poses(cliped_poses)
    return cliped_motion, frame_size


def stitch_motion(motion1, motion2, transition_length, transition_func):
    aligned_motion, _ = align_motion(motion1, motion2)
    aligned_poses = aligned_motion.get_poses()
    motion1_end_pose = motion1.get_poses()[-1]
    aligned_start_pose = aligned_poses[0]
    pos_diff, rot_diff = get_posture_difference(aligned_start_pose, motion1_end_pose)
    for index in range(transition_length):
        weight = 1 - transition_func(index / (transition_length - 1))
        aligned_pose = aligned_poses[index]
        positions = aligned_pose.take_g_position_list()
        pos_mod = positions + (weight * pos_diff)
        aligned_pose.set_g_position_list(pos_mod)
        rot_mod = aligned_pose.take_l_rotmat_list()
        joint_num = aligned_pose.get_joint_num()
        for joint in range(joint_num):
            rot_diff_vec = utility.log(rot_diff[joint])
            rot_mod[joint] = rot_mod[joint] @ utility.exp(weight * rot_diff_vec)
        aligned_pose.set_l_rotmat_list(rot_mod)
    aligned_motion, new_max_frame = concat_motion(motion1, aligned_motion)
    return aligned_motion, new_max_frame


def blend_motion(motion1, motion2, motion1_segment_start, motion2_segment_start,
                 motion1_segment_size, motion2_segment_size,
                 blending_length, blending_func):
    motion1_len = motion1.get_max_frame()
    motion2_len = motion2.get_max_frame()
    if motion1_segment_start + motion1_segment_size > motion1_len:
        motion1_segment_end = motion1.get_max_frame()
    else:
        motion1_segment_end = motion1_segment_start + motion1_segment_size
    if motion2_segment_start + motion2_segment_size > motion2_len:
        motion2_segment_end = motion2.get_max_frame()
    else:
        motion2_segment_end = motion2_segment_start + motion2_segment_size
    first_motion = copy.deepcopy(motion1)
    cliped_motion1 = copy.deepcopy(first_motion)
    cliped_motion1, cliped_size1 = clip_motion(cliped_motion1, motion1_segment_start, motion1_segment_end, "_clip1")

    second_motion = copy.deepcopy(motion2)
    cliped_motion2 = copy.deepcopy(second_motion)
    cliped_motion2, cliped_size2 = clip_motion(cliped_motion2, motion2_segment_start, motion2_segment_end, "_clip2")

    first_motion, first_motion_size = clip_motion(first_motion, 0, motion1_segment_start, "_without_clip1")
    second_motion, second_motion_size = clip_motion(second_motion, motion2_segment_end, second_motion.get_max_frame())

    # uniform resampling
    '''
    def clip1_interpolation_func(t):
        return ((cliped_size1-1)/(blending_length)) * t
    def clip2_interpolation_func(t):
        return ((cliped_size2-1)/(blending_length)) * t
    '''
    # non-uniform resampling
    def clip1_interpolation_func(t):
        f1 = 3/2
        f2 = 3/4
        t = t/(blending_length-1)
        s = -2*t*t*t + 3*t*t + f1*(t*t*t - 2*t*t + t) + f2*(t*t*t - t*t)
        return (cliped_size1 -1) * s

    def clip2_interpolation_func(t):
        f1 = 2 / 3
        f2 = 4 / 3
        t = t / (blending_length-1)
        s = -2 * t * t * t + 3 * t * t + f1 * (t * t * t - 2 * t * t + t) + f2 * (t * t * t - t * t)
        return (cliped_size2 - 1) * s

    cliped_motion1, warped_size1 = warp_time(cliped_motion1, clip1_interpolation_func)
    cliped_motion2, warped_size2 = warp_time(cliped_motion2, clip2_interpolation_func)


    aligned_clip, _ = align_motion(cliped_motion1, cliped_motion2, 0, 0)
    cliped_motion2 = aligned_clip

    cliped_poses1 = cliped_motion1.get_poses()
    cliped_poses2 = cliped_motion2.get_poses()

    blended_poses = []
    for index in range(blending_length):
        t = index/(blending_length-1)
        weight = blending_func(t)
        pose1 = cliped_poses1[index]
        pose2 = cliped_poses2[index]
        blended_pose = interpolate_posture(pose1, pose2, weight)
        blended_poses.append(blended_pose)
    blended_clip = cliped_motion1
    blended_clip.set_max_frame(blending_length)
    blended_clip.set_name(motion1.get_name() + "_" + motion2.get_name() + "_blended")
    blended_clip.set_poses(blended_poses)

    # semi_blended_motion, _ = stitch_motion(first_motion, blended_clip, 10, blending_func)
    # blended_motion, _ = stitch_motion(semi_blended_motion, second_motion, 10, blending_func)

    semi_blended_motion, _ = concat_motion(first_motion, blended_clip)
    second_motion, _ = align_motion(semi_blended_motion, second_motion)
    blended_motion, _ = concat_motion(semi_blended_motion, second_motion)

    # blended_motion = blended_clip

    return blended_motion
