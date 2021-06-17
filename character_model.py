import numpy as np
from abc import *

import utility
import kinematics
import motion_utility


class Character:
    def __init__(self):
        super().__init__()
        self.skeleton = None
        self.motion = None

        self.model_origin = np.array([0, 0, 0])
        self.model_orientation_rotmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.model_scale = 1

    def make_new_skeleton(self, name):
        self.skeleton = Skeleton(name)
        return self.skeleton

    def make_new_motion(self, name):
        self.motion = Motion(name)
        return self.motion

    def set_model_scale(self, scale):
        self.model_scale = scale

    def get_model_scale(self):
        return self.model_scale

    def set_model_origin(self, location):
        self.model_origin = location

    def get_model_origin(self):
        return self.model_origin

    def set_model_orientation(self, rotamt):
        self.model_orientation_rotmat = rotamt

    def get_model_orientation(self):
        return self.model_orientation_rotmat

    def set_skeleton(self, skeleton):
        self.skeleton = skeleton

    def get_skeleton(self):
        return self.skeleton

    def set_motion(self, motion):
        self.motion = motion

    def get_motion(self):
        return self.motion

    def get_max_frame(self):
        return self.motion.get_max_frame()

    def get_offsets(self):
        offset_list = self.skeleton.get_offsets()
        return offset_list

    def get_kintrees(self):
        parentree, childree = self.skeleton.get_kintrees()
        return parentree, childree

    def get_motion_info(self, frame):
        parentree, _ = self.get_kintrees()
        offset_list = self.get_offsets()
        rotmats, positions, velocities = self.motion.get_pose_info(frame, parentree, offset_list)
        return rotmats, positions, velocities

    def update_motion_item(self, frame, value):
        self.motion.update_pos(frame, value)

    def time_warp(self, scale_function):
        warped_motion, new_max_frame = motion_utility.warp_time(self.motion, scale_function)
        self.motion = warped_motion
        return new_max_frame

    def motion_warp(self, keyframe, target_character, start_delta, end_delta, weight_func):
        target_motion = target_character.get_motion()
        keyframe_posture = target_motion.get_poses()[keyframe]
        warped_motion = motion_utility.warp_motion(self.motion, keyframe, keyframe_posture,
                                                   start_delta, end_delta, weight_func)
        self.motion = warped_motion

    def motion_clip(self, start_frame, size, clip_name=""):
        if start_frame < 0:
            start_frame = 0
        end_frame = start_frame + size
        if end_frame >= self.get_max_frame():
            end_frame = self.get_max_frame()
        motion = self.get_motion()
        cliped_motion, cliped_size = motion_utility.clip_motion(motion, start_frame, end_frame, clip_name)
        self.set_motion(cliped_motion)
        return cliped_size

    def motion_stitch(self, character1, character2, transition_length, transition_func):
        motion1 = character1.get_motion()
        motion2 = character2.get_motion()
        stitched_motion, new_max_frame = motion_utility.stitch_motion(motion1, motion2,
                                                                      transition_length, transition_func)
        self.motion = stitched_motion
        return new_max_frame

    def motion_blend(self, character1, character2, motion1_seg_start, motion2_seg_start,
                     motion1_seg_size, motion2_seg_size, blending_length, blending_func):
        motion1 = character1.get_motion()
        motion2 = character2.get_motion()
        blended_motion = motion_utility.blend_motion(motion1, motion2, motion1_seg_start, motion2_seg_start,
                                                     motion1_seg_size, motion2_seg_size, blending_length, blending_func)
        self.motion = blended_motion
        max_frame = blended_motion.get_max_frame()
        return max_frame

    def motion_align(self, character1, character2):
        motion1 = character1.get_motion()
        motion2 = character2.get_motion()
        aligned_motion, max_frame = motion_utility.align_motion(motion1, motion2)

        self.motion = aligned_motion
        return max_frame


class Skeleton:
    def __init__(self, name):
        self.skeleton_name = name
        self.joint_num = 0       # joint 개수
        self.joint_list = []     # joint list
        self.joint_parentree = []  # joint kintree parent index list
        self.joint_childree = {}    # joint kintree child index list
        self.end_joint_id_list = []      # end_joint_id_list

    def stack_joint(self, joint):
        self.joint_list.append(joint)
        self.joint_num += 1

    def make_new_joint(self, is_end=False):
        new_joint = Joint()
        if is_end:
            self.stack_end_joint_id()
        self.stack_joint(new_joint)
        return new_joint

    def get_joint(self, joint_id):
        return self.joint_list[joint_id]

    def get_joint_id(self, me):
        return self.joint_list.index(me)

    def stack_parent(self, parent_id):
        self.joint_parentree.append(parent_id)

    def get_parent_id(self, me):
        my_id = self.get_joint_id(me)
        parent_id = self.joint_parentree[my_id]
        return parent_id

    def update_children(self, parent_id, child_id):
        if parent_id in self.joint_childree:
            children = self.joint_childree[parent_id]
            children.append(child_id)
            self.joint_childree.update({parent_id: children})
        else:
            self.joint_childree.update({parent_id: [child_id]})

    def get_children_id(self, me):
        my_id = self.get_joint_id(me)
        children_id = self.joint_childree[my_id]
        return children_id

    def stack_end_joint_id(self):
        self.end_joint_id_list.append(self.joint_num)

    def get_joint_num(self):
        return self.joint_num

    def get_end_list(self):
        return self.end_joint_id_list

    def get_name(self):
        return self.skeleton_name

    def get_kintrees(self):
        return self.joint_parentree, self.joint_childree

    def get_offsets(self):
        offset_list = []
        for joint in self.joint_list:
            offset_list.append(joint.get_offset())
        return offset_list


class Joint:
    def __init__(self):
        self.name = ''  # joint name
        self.offset = []         # offset

    def set_name(self, name):
        self.name = name

    def set_offset(self, offset):
        self.offset = offset

    def get_name(self):
        return self.name

    def get_offset(self):
        return self.offset


class Motion:
    def __init__(self, name):
        self.motion_name = name
        self.max_frame = 0          # frame 수
        self.poses = []         # pose list [ {pose object} x Frame ]
        self.fps = 0

    def set_fps(self, fps):
        self.fps = fps

    def get_fps(self):
        return self.fps

    def set_max_frame(self, max_frame):
        self.max_frame = max_frame

    def get_max_frame(self):
        return self.max_frame

    def get_name(self):
        return self.motion_name

    def set_name(self, name):
        self.motion_name = name

    def stack_pose(self, pose):
        self.poses.append(pose)
        self.max_frame += 1

    def make_new_pose(self):
        pose = Pose()
        self.stack_pose(pose)
        return pose

    def del_pose(self, frame):
        pop_out = self.poses.pop(frame)
        self.max_frame -= 1
        return pop_out

    def get_poses(self):
        return self.poses

    def set_poses(self, pose_list):
        self.poses = pose_list
        self.max_frame = len(self.poses)

    def get_pose_info(self, frame, parentree, offset_list):
        if frame > self.max_frame:
            frame = self.max_frame
        if frame > 0:
            before_pose = self.poses[frame-1]
        else:
            before_pose = self.poses[frame]
        return self.poses[frame].take_l_rotmat_list(),\
               self.poses[frame].get_g_position_list(parentree, offset_list),\
               self.poses[frame].get_g_velocity_list(parentree, offset_list, before_pose, self.fps)

    def update_pos(self, frame, value):
        self.poses[frame].set_l_rotmat_list(value)


class Pose:
    def __init__(self):
        self.local_coor_rotmat_list = np.array([])
        self.global_position_list = np.array([])
        self.global_velocity_list = np.array([])
        self.rotation_changed = False
        self.position_changed = False

    def get_joint_num(self):
        return self.local_coor_rotmat_list.shape[0]

    def set_l_rotmat_list(self, rotmat_list):
        self.rotation_changed = True
        self.local_coor_rotmat_list = rotmat_list

    def update_g_position(self, parentree, offset_list, joint_num=-1):
        self.position_changed = True
        if joint_num == -1:
            joint_num = self.get_joint_num()
        for joint_index in range(joint_num):
            FK_rot_list = [self.local_coor_rotmat_list[joint_index]]
            FK_trans_list = [offset_list[joint_index]]
            parent = parentree[joint_index]
            while parent > -1:
                FK_rot_list.append(self.local_coor_rotmat_list[parent])
                FK_trans_list.append(offset_list[parent])
                parent = parentree[parent]
            FK_rot_list = list(reversed(FK_rot_list))
            FK_trans_list = list(reversed(FK_trans_list))
            global_position = kinematics.FK(FK_rot_list, FK_trans_list, self.global_position_list[0])
            self.set_g_position(joint_index, global_position)
        return self.global_position_list

    def set_g_position_list(self, positions):
        self.position_changed = True
        self.global_position_list = positions

        # init global_velocity_list
        self.global_velocity_list = np.zeros_like(self.global_position_list, dtype=float)

    def set_g_position(self, joint_index, position):
        self.position_changed = True
        self.global_position_list[joint_index] = position

    def take_l_rotmat_list(self):
        return self.local_coor_rotmat_list

    def get_g_position_list(self, parentree, offset_list):
        if self.rotation_changed:
            self.update_g_position(parentree, offset_list)
            self.rotation_changed = False
        return self.take_g_position_list()

    def take_g_position_list(self):
        return self.global_position_list

    def get_g_velocity_list(self, parentree, offset_list, before_pose, fps):
        if self.position_changed:
            self.update_g_velocity(parentree, offset_list, before_pose, fps)
            self.position_changed = False
        return self.global_velocity_list

    def set_g_velocity_list(self, velocities):
        self.global_velocity_list = velocities

    def set_g_velocity(self, joint_index, velocity):
        self.global_velocity_list[joint_index] = velocity

    def update_g_velocity(self, parentree, offset_list, before_pose, fps):
        now_global_positions = self.get_g_position_list(parentree, offset_list)
        before_global_positons = before_pose.get_g_position_list(parentree, offset_list)
        for joint_id in range(len(now_global_positions)):
            global_velocity_per_joint = (now_global_positions[joint_id] - before_global_positons[joint_id]) * fps
            self.set_g_velocity(joint_id, global_velocity_per_joint)