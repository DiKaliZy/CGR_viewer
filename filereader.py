from abc import *
import numpy as np

import character_model
import utility


def check_file_type(file_root):
    temp = 0
    index = 0
    file_type = ''
    file_name = ''
    file_path = ''

    # 파일명, 확장자 확인
    for char in reversed(file_root):
        # 확장자
        if char == '.':
            file_type = file_root[-index:]
            temp = index
        # 파일명
        if char == '/':
            file_name = file_root[(-index):-(temp + 1)]
            file_path = file_root[:-index]
            break
        index = index + 1

    if file_type == 'bvh':
        file_reader = BvhReader(file_root, file_name, file_path)

    else:
        return file_type, None

    return file_type, file_reader


class FileReader(metaclass=ABCMeta):
    def __init__(self, file_root, file_name, file_path):
        self.file_root = file_root
        self.file_name = file_name
        self.file_path = file_path

    @abstractmethod
    def check_file(self):
        pass

    @abstractmethod
    def read_file(self):
        pass

    @abstractmethod
    def make_character(self, data):
        pass


def align_bvh_line(input_data, channels):
    R = np.identity(3)
    P = [0., 0., 0.]
    i = 0
    for order in channels:
        if order == "Xrotation" or order == "XROTATION":
            Rx = utility.get_rot_mat_from([1, 0, 0], np.radians(input_data[i]))
            R = R @ Rx
        elif order == "Yrotation" or order == "YROTATION":
            Ry = utility.get_rot_mat_from([0, 1, 0], np.radians(input_data[i]))
            R = R @ Ry
        elif order == "Zrotation" or order == "ZROTATION":
            Rz = utility.get_rot_mat_from([0, 0, 1], np.radians(input_data[i]))
            R = R @ Rz
        elif order == "Xposition" or order == "XPOSITION":
            P[0] = input_data[i]
        elif order == "Yposition" or order == "YPOSITION":
            P[1] = input_data[i]
        elif order == "Zposition" or order == "ZPOSITION":
            P[2] = input_data[i]
        i += 1
    return R, np.array(P)


class BvhReader(FileReader):
    def __init__(self, file_root, file_name, file_path):
        super().__init__(file_root, file_name, file_path)

    def check_file(self):
        new_character = self.read_file()
        if new_character is None:
            print(self.file_name + " has problem.")
        return new_character

    def read_file(self):
        file = open(self.file_root, 'r')
        lines = file.readlines()
        new_character = self.make_character(lines)
        return new_character

    def make_character(self, lines):
        new_character = character_model.Character()
        is_it_hierarchy = True

        now_joint = None
        old_joint = None

        new_skeleton = new_character.make_new_skeleton(self.file_name)
        new_motion = new_character.make_new_motion(self.file_name)

        joint_channels = []

        for line in lines:
            line = line.strip()
            line = line.replace(":", "")
            line = line.split()

            if line[0] == "MOTION":
                is_it_hierarchy = False

            if is_it_hierarchy:
                if line[0] == "JOINT":
                    new_joint = new_skeleton.make_new_joint()
                    new_joint.set_name(line[1])
                    old_joint = now_joint
                    now_joint = new_joint

                elif line[0] == "ROOT":
                    new_joint = new_skeleton.make_new_joint()
                    new_joint.set_name(line[1])
                    now_joint = new_joint
                    old_joint = now_joint

                elif line[0] == "OFFSET":
                    offset = list(map(float, line[1:4]))  # str array 형태를 float array로 변경
                    if offset[1] > 1.5:
                        new_character.set_model_scale = (1 / 16)
                    now_joint.set_offset(np.array(offset))

                elif line[0] == "CHANNELS":
                    channels = int(line[1])
                    joint_channels.append(line[2:2+channels])

                elif line[0] == "End":
                    new_joint = new_skeleton.make_new_joint(is_end=True)
                    parent = old_joint
                    new_joint.set_name('End_of_' + parent.get_name())
                    joint_channels.append([])
                    old_joint = now_joint
                    now_joint = new_joint

                elif line[0] == "{":
                    if new_skeleton.get_joint_num() > 1:
                        parent_id = new_skeleton.get_joint_id(old_joint)
                        new_skeleton.stack_parent(parent_id)
                        new_skeleton.update_children(parent_id, new_skeleton.get_joint_id(now_joint))

                    else:
                        new_skeleton.stack_parent(-1)

                elif line[0] == "}":
                    now_joint = old_joint
                    old_joint = new_skeleton.get_joint(new_skeleton.get_parent_id(old_joint))

                elif line[0] == "HIERARCHY":
                    pass
                else:
                    return None

            # motion
            elif is_it_hierarchy == False:
                if line[0] == "Frames":
                    # new_motion.set_max_frame(int(line[1]))
                    pass

                elif line[0] == "Frame":
                    new_motion.set_fps(float(line[2]) * 1000)
                elif line[0] == "MOTION":
                    pass
                else:
                    motion_per_frame = list(map(float, line))
                    joint_num = new_skeleton.get_joint_num()
                    rotmat, position = self.read_rotang_list(joint_num, motion_per_frame, joint_channels, new_skeleton)

                    new_pose = new_motion.make_new_pose()
                    new_pose.set_l_rotmat_list(rotmat)
                    new_pose.set_g_position_list(position)

        return new_character

    def read_rotang_list(self, joint_num, line_list, joint_channels, skeleton):
        total_rotmat = []
        total_pos = []
        for index in range(joint_num):
            temp_list = []
            if index in skeleton.get_end_list():
                rotmat = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                total_rotmat.append(rotmat)
                total_pos.append([0., 0., 0.])
            else:
                channels = joint_channels[index]
                for channel in range(len(channels)):
                    temp_list.append(line_list.pop(0))
                R, P = align_bvh_line(temp_list, channels)
                total_rotmat.append(R)
                total_pos.append(P)

        return np.array(total_rotmat), np.array(total_pos)
