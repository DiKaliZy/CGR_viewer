import numpy as np

import filereader
import data_manager
import kinematics


class Messenger:
    def __init__(self):
        super().__init__()
        self.character_manager = data_manager.MotionDataManager()
        self.limb_IK_manager = data_manager.IKMotionDataManager()
        self.jacobian_IK_manager = data_manager.IKMotionDataManager()
        self.particle_manager = data_manager.ParticleDataManager()
        self.play_slider = None
        self.frame_typer = None
        self.start_button = None
        self.joint_button = None
        self.velocity_button = None
        self.end_frame = None
        self.start_frame = None
        self.canvas = None

        # self.jacob_IK_module = kinematics.Jacobian_IK()
        self.jacob_IK_module = kinematics.Jacobian_IK_3axises()

    def make_IK_character(self, character_id):
        target_dict = self.limb_IK_manager.load_data()
        if character_id in target_dict:
            pass
        else:
            self.copy_character_dict(self.limb_IK_manager, character_id)
            self.limb_IK_manager.init_IK_dict(character_id)
        target_dict = self.jacobian_IK_manager.load_data()
        if character_id in target_dict:
            pass
        else:
            self.copy_character_dict(self.jacobian_IK_manager, character_id)
            self.jacobian_IK_manager.init_IK_dict(character_id)

    def copy_character_dict(self, target_manager, character_id):
        self.character_manager.copy_data(target_manager, character_id)

    def read_file(self, filename):
        for file_root in filename:
            file_type, file_reader = filereader.check_file_type(file_root)
            character = file_reader.read_file()
            character_id = self.character_manager.save_data(character)
            self.character_manager.update_character_dict_item(character_id, "file_type", file_type)

    def save_character(self, character_object):
        self.character_manager.save_data(character_object)

    def load_character_info(self, character_id):
        return self.character_manager.get_character_dict_item(character_id, "character")

    def set_slider(self, slider):
        self.play_slider = slider

    def set_canvas(self, canvas):
        self.canvas = canvas

    def set_start_button(self, button):
        self.start_button = button

    def set_joint_button(self, button):
        self.joint_button = button

    def set_velocity_button(self, button):
        self.velocity_button = button

    def set_frame_typer(self, typer):
        self.frame_typer = typer

    def set_end_frame(self, end_frame):
        self.end_frame = end_frame

    def set_start_frame(self, start_frame):
        self.start_frame = start_frame

    def get_character_id_list(self):
        return self.character_manager.load_id_list()

    def get_now_frame(self, character_id):
        return self.character_manager.get_character_dict_item(character_id, "now_frame")

    def update_view_frame_state(self, value):
        self.play_slider.SetValue(value)
        self.frame_typer.SetValue(str(value))
        self.play_slider.Refresh()

    def update_view_base_state(self, character_id):
        max_value = self.character_manager.get_max_frame(character_id) - 1
        end_value = self.character_manager.get_character_dict_item(character_id, "end_frame")
        start_value = self.character_manager.get_character_dict_item(character_id, "start_frame")
        self.play_slider.SetMax(max_value)
        self.end_frame.SetValue(str(end_value))
        self.start_frame.SetValue(str(start_value))
        # self.play_slider.Refresh()

    def get_character_states(self, character_id):
        box = {}
        box.update({"now_frame": self.character_manager.get_character_dict_item(character_id, "now_frame")})
        box.update({"init_loaded": self.character_manager.get_character_dict_item(character_id, "init_loaded")})
        box.update({"focused": self.character_manager.get_character_dict_item(character_id, "focused")})
        box.update({"pinned": self.character_manager.get_character_dict_item(character_id, "pinned")})
        scale, parentree, childree, offset_list = self.character_manager.load_character_for_draw(character_id)
        box.update({"scale": scale})
        box.update({"parentree": parentree})
        box.update({"childree": childree})
        box.update({"offset_list": offset_list})
        box.update({"moded_type": self.character_manager.get_character_dict_item(character_id, "moded_type")})
        return box

    def get_character_motion(self, character_id, frame, category="character"):
        if category == "character":
            rotmats, positions, velocities = self.character_manager.load_motion_for_draw(character_id, frame)
        elif category == "limb":
            rotmats, positions, velocities = self.limb_IK_manager.load_motion_for_draw(character_id, frame)
        else:
            rotmats, positions, velocities = self.jacobian_IK_manager.load_motion_for_draw(character_id, frame)
        return rotmats, positions, velocities

    def update_frame(self, character_id, value, is_circulate=False):
        start_frame = self.character_manager.get_character_dict_item(character_id, "start_frame")
        end_frame = self.character_manager.get_character_dict_item(character_id, "end_frame")
        if end_frame >= start_frame:
            if value > end_frame:
                if is_circulate:
                    value = start_frame
                else:
                    value = end_frame
            elif value < start_frame:
                if is_circulate:
                    value = end_frame
                else:
                    value = start_frame
        else:
            if end_frame < value < start_frame:
                end_relate = value - end_frame
                start_relate = start_frame - value
                if is_circulate:
                    value = start_frame
                else:
                    if end_relate > start_relate:
                        value = start_frame
                    else:
                        value = end_frame

        self.character_manager.update_character_dict_item(character_id, "now_frame", value)

        return value

    def change_frame(self, value, is_relative_value=False, allow_pinned=False):
        id_list = list(self.character_manager.load_id_list())
        for character_id in id_list:
            is_focused = self.character_manager.get_character_dict_item(character_id, "focused")
            if is_relative_value:
                now_frame = self.get_now_frame(character_id)
                value = now_frame + value
            if is_focused:
                value = self.update_frame(character_id, value)
                self.update_view_frame_state(value)
            elif allow_pinned:
                is_pinned = self.character_manager.get_character_dict_item(character_id, "pinned")
                if is_pinned:
                    value = self.update_frame(character_id, value)
        return value

    def play_frame(self):
        id_list = list(self.character_manager.load_id_list())
        for character_id in id_list:
            is_played = self.character_manager.get_character_dict_item(character_id, "played")
            is_focused = self.character_manager.get_character_dict_item(character_id, "focused")
            is_pinned = self.character_manager.get_character_dict_item(character_id, "pinned")
            if is_focused or is_pinned:
                now_frame = self.get_now_frame(character_id) + 1
                if is_played:
                    now_frame = self.update_frame(character_id, now_frame, is_circulate=True)
                    if is_focused:
                        self.update_view_base_state(character_id)
                        self.update_view_frame_state(now_frame)

    def change_play_state(self):
        id_list = list(self.character_manager.load_id_list())
        id_list.sort()
        focused_play_state = False
        for character_id in id_list:
            is_focused = self.character_manager.get_character_dict_item(character_id, "focused")
            is_pinnded = self.character_manager.get_character_dict_item(character_id, "pinned")
            is_init_loaded = self.character_manager.get_character_dict_item(character_id, "init_loaded")
            is_played = self.character_manager.get_character_dict_item(character_id, "played")
            if is_focused or is_pinnded:
                if is_init_loaded:
                    self.character_manager.update_character_dict_item(character_id, "init_loaded", False)
            if is_focused:
                if is_played:
                    focused_play_state = False
                    self.character_manager.update_character_dict_item(character_id, "played", False)
                else:
                    focused_play_state = True
                    self.character_manager.update_character_dict_item(character_id, "played", True)
            elif is_pinnded:
                self.character_manager.update_character_dict_item(character_id, "played", focused_play_state)

        return focused_play_state

    def change_frame_state(self, target, value):
        id_list = list(self.character_manager.load_id_list())
        id_list.sort()
        for character_id in id_list:
            is_focused = self.character_manager.get_character_dict_item(character_id, "focused")
            if is_focused:
                max_frame = self.character_manager.get_max_frame(character_id)
                if value >= max_frame:
                    value = max_frame - 1
                elif value < 0:
                    value = 0
                self.character_manager.update_character_dict_item(character_id, target, value)
        return value

    def change_joint_view_state(self):
        state = self.canvas.change_property_view_state("joint_view")
        return state

    def change_velocity_view_state(self):
        state = self.canvas.change_property_view_state("velocity_view")
        return state

    def change_joint_button(self, state):
        self.joint_button.change_label(state)

    def change_velocity_button(self, state):
        self.velocity_button.change_label(state)

    def change_play_button(self, state):
        self.start_button.change_label(state)

    def get_global_rotmat_list(self, target_manager, character_id, frame):
        return target_manager.get_global_rotmat_list(character_id, frame)

    def get_IK_keyframes(self, character_id):
        keyframes = self.limb_IK_manager.get_IK_keyframes(character_id)
        return keyframes

    def check_IK(self, character_id, now_frame, category="limb"):
        if category == "limb":
            IK_check, history_dict = self.limb_IK_manager.check_IK(character_id, now_frame)
        else:
            IK_check, history_dict = self.jacobian_IK_manager.check_IK(character_id, now_frame)

        return IK_check, history_dict

    def get_limb_IK_rot_mat(self, target_index, target_position, parent_list, g_pos_list, rot_list,
                            global_mats):
        parent = parent_list[target_index]
        grandparent = parent_list[parent]
        a_pos = g_pos_list[grandparent]
        b_pos = g_pos_list[parent]
        c_pos = g_pos_list[target_index]
        ori_a_rot_mat = rot_list[grandparent]
        ori_b_rot_mat = rot_list[parent]
        a_global_mat = global_mats[grandparent]
        b_global_mat = global_mats[parent]
        changed_a_rotmat, changed_b_rotmat = \
            kinematics.limb_IK(target_position, a_pos, b_pos, c_pos,
                                        ori_a_rot_mat, ori_b_rot_mat, a_global_mat, b_global_mat)
        return changed_a_rotmat, changed_b_rotmat

    def calcul_limb_IK(self, character_id, frame, target_index, target_position):
        _, parent_list, _, _ = self.limb_IK_manager.load_character_for_draw(character_id)
        rot_list, g_pos_list, _ = self.limb_IK_manager.load_motion_for_draw(character_id, frame)
        global_mats = self.get_global_rotmat_list(self.limb_IK_manager, character_id, frame)
        changed_a, changed_b = self.get_limb_IK_rot_mat(target_index, target_position,
                                                        parent_list, g_pos_list, rot_list, global_mats)
        parent = parent_list[target_index]
        grand_parent = parent_list[parent]
        rot_list[parent] = changed_b
        rot_list[grand_parent] = changed_a
        self.limb_IK_manager.update_motion_item(character_id, frame, rot_list)

    def get_jacob_IK_rot_mat(self, desired_position, original_rotmats, original_positions, offsets,
                             threshold, max_iter, beta):
        changed_rotmats = \
            self.jacob_IK_module.jacobian_IK(desired_position, original_rotmats, original_positions, offsets,
                                             threshold, max_iter, beta)
        return changed_rotmats

    def calcul_jacob_IK(self, character_id, frame, target_index, desired_position,
                        threshold=0.005, max_iter=800, beta=0.05):
        _, parent_list, _, offset_list = self.jacobian_IK_manager.load_character_for_draw(character_id)
        original_rotmat_list, original_position_list, _ = \
            self.jacobian_IK_manager.load_motion_for_draw(character_id, frame)

        target2root_joint_id_list = [target_index]
        target2root_rot_list = [original_rotmat_list[target_index]]
        target2root_pos_list = [original_position_list[target_index]]
        target2root_offset_list = [offset_list[target_index]]
        parent = parent_list[target_index]
        while parent > -1:
            target2root_joint_id_list.append(parent)
            target2root_rot_list.append(original_rotmat_list[parent])
            target2root_pos_list.append(original_position_list[parent])
            target2root_offset_list.append(offset_list[parent])
            parent = parent_list[parent]
        root2target_joint_id_list = list(reversed(target2root_joint_id_list))
        root2target_rot_list = np.array(list(reversed(target2root_rot_list)))
        root2target_pos_list = np.array(list(reversed(target2root_pos_list)))
        root2target_offset_list = np.array(list(reversed(target2root_offset_list)))

        changed_rotmats = \
            self.get_jacob_IK_rot_mat(desired_position, root2target_rot_list, root2target_pos_list,
                                      root2target_offset_list, threshold, max_iter, beta)
        for index in range(len(changed_rotmats)):
            original_rotmat_list[root2target_joint_id_list[index]] = changed_rotmats[index]
        self.jacobian_IK_manager.update_motion_item(character_id, frame, original_rotmat_list)

    def call_IK_mod(self, selected_character_id, selected_joint_id, x_mod=0, y_mod=0, z_mod=0):
        delta_vector = np.array([x_mod, y_mod, z_mod], dtype=float)
        now_frame = self.character_manager.get_character_dict_item(selected_character_id, "now_frame")
        self.make_IK_character(selected_character_id)

        desired_position = self.limb_IK_manager.get_desired_position(selected_character_id,
                                                                     selected_joint_id, now_frame, delta_vector)
        self.limb_IK_manager.update_IK_dict(selected_character_id, now_frame, selected_joint_id, desired_position)
        self.calcul_limb_IK(selected_character_id, now_frame, selected_joint_id, desired_position)

        desired_position = self.jacobian_IK_manager.get_desired_position(selected_character_id,
                                                                         selected_joint_id, now_frame, delta_vector)
        self.jacobian_IK_manager.update_IK_dict(selected_character_id, now_frame, selected_joint_id, desired_position)
        # self.calcul_jacob_IK(selected_character_id, now_frame, selected_joint_id, desired_position)

    def time_warp_character(self, character_id, func):
        self.character_manager.time_warp_character(character_id, func)

    def motion_warp_character(self, character_id, keyframe_list, start_delta, end_delta, func):
        self.character_manager.motion_warp_character(character_id, keyframe_list, self.limb_IK_manager,
                                                     start_delta, end_delta, func)

    def motion_clip_character(self, character_id, clip_start_frame, clip_size):
        clipped_character_id = self.character_manager.motion_clip_character(character_id, clip_start_frame, clip_size)
        return clipped_character_id

    def motion_stitch_character(self, character1_id, clip1_start_frame, clip1_size,
                                character2_id, clip2_start_frame, clip2_size,
                                transtion_length, func):
        clipped_id1 = self.motion_clip_character(character1_id, clip1_start_frame, clip1_size)
        clipped_id2 = self.motion_clip_character(character2_id, clip2_start_frame, clip2_size)
        self.character_manager.motion_stitch_character(clipped_id1, clipped_id2, transtion_length, func)
        self.character_manager.motion_align_character(clipped_id1, clipped_id2)

    def motion_blend_character(self, character1_id, character2_id, motion1_seg_start, motion2_seg_start,
                               motion1_seg_size, motion2_seg_size, blending_size, func):
        self.character_manager.motion_blend_character(character1_id, character2_id,
                                                      motion1_seg_start, motion2_seg_start,
                                                      motion1_seg_size, motion2_seg_size, blending_size, func)

    def make_test_particle(self, position, velocity, mass):
        particle = self.particle_manager.make_new_particle(position, velocity, mass)
        return particle

    def test_particles(self, time_step, iter_num):
        for i in range(iter_num):
            self.particle_manager.get_next_step(time_step)

    def get_test_particle(self, id):
        particle = self.particle_manager.get_particle(id)
        return particle

    def make_test_spring(self, particle1, particle2):
        self.particle_manager.make_spring_connection(particle1, particle2)

    def get_springs(self):
        return self.particle_manager.get_springs()
