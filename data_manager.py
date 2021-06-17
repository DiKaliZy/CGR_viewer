from abc import *
import copy
import numpy as np

import messenger
import character_model
import dynamics_class
import utility
import physics


class DataManager(metaclass=ABCMeta):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def save_data(self, data):
        pass


class MotionDataManager(DataManager):
    def __init__(self):
        # 구조 :
        # {id: {play 여부, focus 여부, init_load 여부, 현재 frame, 구간반복 시작 frame, 구간반복 종료 frame,
        # character object, file_type}}
        self.character_dict = {}
        self.character_dict_format = dict(played=False, focused=True, pinned=False, init_loaded=True,
                                          now_frame=0, start_frame=0, end_frame=0,
                                          character=None, file_type='', moded_type="original", origin_id=[-1])
        self.character_numbering = 0

    def get_character_num(self):
        return len(self.character_dict)

    def load_data(self):
        return self.character_dict

    def copy_data(self, target_manager, character_id):
        dict_content = copy.deepcopy(self.character_dict[character_id])
        target_manager.set_dict_item(character_id, dict_content)

    def set_dict_item(self, character_id, dict_content):
        if character_id in self.character_dict:
            pass
        else:
            self.character_numbering += 1
        self.character_dict.update({character_id: dict_content})

    def save_data(self, character_object):
        character_id = self.character_numbering
        dict_content = copy.deepcopy(self.character_dict_format)
        dict_content.update({"end_frame": character_object.get_max_frame()-1, "character": character_object})
        self.set_dict_item(character_id, dict_content)
        return character_id

    def load_character_for_draw(self, character_id):
        character = self.get_character_dict_item(character_id, "character")
        model_scale = character.get_model_scale()
        parentree, childree = character.get_kintrees()
        offset_list = character.get_offsets()

        return model_scale, parentree, childree, offset_list

    def load_motion_for_draw(self, character_id, frame):
        character = self.get_character_dict_item(character_id, "character")
        rotmats, positions, velocities = character.get_motion_info(frame)
        return rotmats, positions, velocities

    def load_id_list(self):
        return self.character_dict.keys()

    def update_motion_item(self, character_id, frame, value):
        character = self.get_character_dict_item(character_id, "character")
        character.update_motion_item(frame, value)

    def get_character_dict_help(self):
        return self.character_dict_format.keys()

    def get_character_dict_item(self, character_id, item_name):
        return self.character_dict[character_id][item_name]

    def update_character_dict_item(self, character_id, item_name, value):
        item = self.character_dict[character_id]
        item.update({item_name: value})
        self.character_dict.update({character_id: item})

    def get_max_frame(self, character_id):
        character = self.get_character_dict_item(character_id, "character")
        max_frame = character.get_max_frame()
        return max_frame

    def get_global_rotmat_list(self, character_id, frame):
        character = self.get_character_dict_item(character_id, "character")
        parentree, childree = character.get_kintrees()
        rotmats, _, _ = character.get_motion_info(frame)
        global_rotmat_list = []
        for index in range(len(parentree)):
            parent = parentree[index]
            g_transform = rotmats[index]
            while parent > -1:
                parent_rotmat = rotmats[parent]
                g_transform = parent_rotmat @ g_transform
                parent = parentree[parent]
            global_rotmat_list.append(g_transform)
        return global_rotmat_list

    def copy_item_n_character(self, character_id):
        origin_dict_item = self.character_dict[character_id]
        moded_dict_item = copy.deepcopy(origin_dict_item)
        origin_character = self.get_character_dict_item(character_id, "character")
        moded_character = copy.deepcopy(origin_character)
        return moded_dict_item, moded_character

    def add_character(self, new_character_id, new_dict_item, new_character, new_max_frame, new_moded_type, origins):
        self.set_dict_item(new_character_id, new_dict_item)
        self.update_character_dict_item(new_character_id, "character", new_character)
        self.update_character_dict_item(new_character_id, "end_frame", new_max_frame)
        self.update_character_dict_item(new_character_id, "moded_type", new_moded_type)
        self.update_character_dict_item(new_character_id, "origin_id", origins)

    def time_warp_character(self, character_id, func):
        moded_dict_item, moded_character = self.copy_item_n_character(character_id)
        new_max_frame = moded_character.time_warp(func)
        new_character_id = character_id + 300
        self.add_character(new_character_id, moded_dict_item, moded_character,
                           new_max_frame - 1, "time_warping", [character_id])

    def motion_warp_character(self, character_id, keyframe_list, IKMotionDataManager, start_delta, end_delta, func):
        moded_dict_item, moded_character = self.copy_item_n_character(character_id)
        for index in range(len(keyframe_list)):
            target_frame = keyframe_list[index]
            target_character = IKMotionDataManager.get_character_dict_item(character_id, "character")
            moded_character.motion_warp(target_frame, target_character, start_delta, end_delta, func)
        new_character_id = character_id + 400
        max_frame = moded_character.get_max_frame()
        self.add_character(new_character_id, moded_dict_item, moded_character, max_frame - 1,
                           "motion_warping", [character_id])

    def motion_clip_character(self, character_id, clip_start_frame, clip_size):
        clipped_dict_item, clipped_character = self.copy_item_n_character(character_id)
        clip_size = clipped_character.motion_clip(clip_start_frame, clip_size, "_clipped")
        clipped_character_id = character_id + 700
        self.add_character(clipped_character_id, clipped_dict_item, clipped_character, clip_size - 1,
                           "motion_clipping", [character_id])
        return clipped_character_id

    def motion_stitch_character(self, character1_id, character2_id, transition_length, func):
        stitched_dict_item, stitched_character = self.copy_item_n_character(character1_id)
        character1 = self.get_character_dict_item(character1_id, "character")
        character2 = self.get_character_dict_item(character2_id, "character")
        new_max_frame = stitched_character.motion_stitch(character1, character2, transition_length, func)
        stitched_character_id = 10000*character2_id + 1000*character1_id + 500
        self.add_character(stitched_character_id, stitched_dict_item, stitched_character, new_max_frame-1,
                           "motion_stitching", [character1_id, character2_id])

    def motion_align_character(self, character1_id, character2_id):
        aligned_dict_item, aligned_character = self.copy_item_n_character(character1_id)
        character1 = self.get_character_dict_item(character1_id, "character")
        character2 = self.get_character_dict_item(character2_id, "character")
        new_max_frame = aligned_character.motion_align(character1, character2)
        aligned_character_id = 30000 * character1_id
        self.add_character(aligned_character_id, aligned_dict_item, aligned_character, new_max_frame-1,
                           "motion_align", [character1_id, character2_id])

    def motion_blend_character(self, character1_id, character2_id, motion1_seg_start, motion2_seg_start,
                               motion1_seg_size, motion2_seg_size, blending_length, func):
        blended_dict_item, blended_character = self.copy_item_n_character(character1_id)
        character1 = self.get_character_dict_item(character1_id, "character")
        character2 = self.get_character_dict_item(character2_id, "character")
        new_max_frame = blended_character.motion_blend(character1, character2, motion1_seg_start, motion2_seg_start,
                                                       motion1_seg_size, motion2_seg_size, blending_length, func)
        blended_character_id = 10000 * character2_id + 1000*character1_id + 600
        self.add_character(blended_character_id, blended_dict_item, blended_character, new_max_frame-1,
                           "motion_blending", [character1_id, character2_id])


class IKMotionDataManager(MotionDataManager):
    def __init__(self):
        super().__init__()
        self.target_history = {}
        self.character_dict_format.update({"IK_history": None})
        self.IK_history_format = dict(frame={"joint_id": np.array([0., 0., 0.])})

    def init_IK_dict(self, character_id):
        character_dict = self.character_dict[character_id]
        character_dict.update({"IK_history": {}})

    def check_IK(self, character_id, frame):
        history_dict = {}
        if character_id in self.character_dict:
            if frame in self.character_dict[character_id]["IK_history"]:
                history_dict = self.get_IK_target_position(character_id, frame)
                return True, history_dict
        return False, history_dict

    def update_IK_dict(self, character_id, frame, joint_id, target_position):
        new_item_joint = {joint_id: target_position}
        new_item_frame = {frame: new_item_joint}

        if character_id in self.character_dict:
            IK_dict = self.character_dict[character_id]["IK_history"]
            if frame in IK_dict:
                joint_dict = IK_dict[frame]
                joint_dict.update(new_item_joint)
            else:
                IK_dict.update(new_item_frame)

    def get_IK_target_position(self, character_id, frame):
        character_IK_dict = self.character_dict[character_id]["IK_history"]
        character_history = character_IK_dict[frame]
        return character_history

    def get_IK_keyframes(self, character_id):
        if character_id in self.character_dict:
            character_IK_dict = self.character_dict[character_id]["IK_history"]
            return list(character_IK_dict.keys())
        else:
            return []

    def get_desired_position(self, selected_character_id, selected_joint_id, now_frame, delta_vector):
        _, positions, _ = self.load_motion_for_draw(selected_character_id, now_frame)
        IK_check, history_dict = self.check_IK(selected_character_id, now_frame)
        if IK_check:
            if selected_joint_id in history_dict:
                now_position = history_dict[selected_joint_id]
            else:
                now_position = positions[selected_joint_id]
        else:
            now_position = positions[selected_joint_id]
        desired_position = now_position + delta_vector
        return desired_position


class ParticleDataManager(DataManager):
    def __init__(self):
        super().__init__()
        self.particle_dict = {}
        self.spring_list = []
        self.numbering = 0

    def load_data(self):
        return self.particle_dict

    def save_data(self, data):
        self.particle_dict.update(data)

    def make_spring_connection(self, particle_1, particle_2):
        spring = dynamics_class.Spring(particle_1, particle_2)
        self.spring_list.append(spring)
        particle_1.make_spring_connection(particle_2, spring)
        particle_2.make_spring_connection(particle_1, spring)

    def make_new_particle(self, position, velocity, mass):
        new_particle = dynamics_class.Particle(position, velocity, mass)
        self.particle_dict.update({self.numbering: new_particle})
        self.numbering += 1
        return new_particle

    def get_next_step(self, time_delta, step_type="euler"):

        def select_step_type(step_type):
            if step_type == "euler":
                step_func = physics.euler_step
            else:
                step_func = physics.mid_point_step
            return step_func

        keys = list(self.particle_dict.keys())
        step_funcs = {}
        funcs = {}
        for key in keys:
            step_funcs.update({key:select_step_type(step_type)})
            # 나중에 derivative evaluation algorithm 변경 필요하면 외부에서 바꿀 수 있도록
            funcs.update({key:physics.deriv_eval})

        new_states, new_times = physics.take_new_state(keys, step_funcs, funcs, time_delta, self.particle_dict)
        print("new_states:", new_states)
        self.update_particle(keys, new_states, new_times)

    def update_particle(self, keys, new_states, new_times):
        for key in keys:
            data = self.particle_dict[key]
            data.update_state(new_states[key])
            data.update_time(new_times[key])

    # TODO: test용 구현부
    def get_particle(self, particle_id):
        return self.particle_dict[particle_id]

    def get_springs(self):
        return self.spring_list

def main():
    motion_data_manager = MotionDataManager()
    print(motion_data_manager.get_character_dict_help())


if __name__ == '__main__':
    main()
