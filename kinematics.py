import numpy as np
import math
import copy

import utility


def limb_IK(target_position, a_pos, b_pos, c_pos,
            original_a_rot_mat, original_b_rot_mat, a_global_mat, b_global_mat):
    def take_limb_IK_step1(a_pos, b_pos, c_pos, t_pos, a_global_mat, b_global_mat):
        ab_vec = b_pos - a_pos
        bc_vec = c_pos - b_pos
        ac_vec = c_pos - a_pos
        at_vec = t_pos - a_pos

        ab_len = utility.l2norm(ab_vec)
        bc_len = utility.l2norm(bc_vec)
        ac_len = utility.l2norm(ac_vec)
        at_len = np.clip(utility.l2norm(at_vec), 0., ab_len + bc_len)

        ap_angle = utility.cos_rule(ab_len, at_len, bc_len)
        bp_angle = utility.cos_rule(ab_len, bc_len, at_len)
        a_angle = utility.cos_rule(ab_len, ac_len, bc_len)
        b_angle = utility.cos_rule(ab_len, bc_len, ac_len)
        a_angle_diff = ap_angle - a_angle
        b_angle_diff = bp_angle - b_angle

        bac_rot_axis = utility.normalized(np.cross(ac_vec, ab_vec))
        a_local_rot_vec = a_global_mat.T @ bac_rot_axis
        b_local_rot_vec = b_global_mat.T @ bac_rot_axis
        a_rot_diff_mat = utility.get_rot_mat_from(a_local_rot_vec, a_angle_diff)
        b_rot_diff_mat = utility.get_rot_mat_from(b_local_rot_vec, b_angle_diff)

        return a_rot_diff_mat, b_rot_diff_mat

    # step 1
    local_a_rot_diff_mat, local_b_rot_diff_mat = take_limb_IK_step1(a_pos, b_pos, c_pos, target_position,
                                                                    a_global_mat, b_global_mat)

    changed_a_rot_mat = original_a_rot_mat @ local_a_rot_diff_mat
    changed_b_rot_mat = original_b_rot_mat @ local_b_rot_diff_mat

    a_g = a_global_mat @ local_a_rot_diff_mat

    def take_limb_IK_step2(a_pos, c_pos, t_pos, a_global_mat):
        ac_vec = c_pos - a_pos
        at_vec = t_pos - a_pos
        ac_unit_vec = utility.normalized(ac_vec)
        at_unit_vec = utility.normalized(at_vec)

        angle = math.acos(np.clip(np.dot(ac_unit_vec, at_unit_vec), -1, 1))

        cat_rot_axis = utility.normalized(np.cross(ac_vec, at_vec))
        local_cat_rot_axis = a_global_mat.T @ cat_rot_axis
        a_rot_diff_mat = utility.get_rot_mat_from(local_cat_rot_axis, angle)

        return a_rot_diff_mat

    # step 2
    local_a_rot_diff_mat = take_limb_IK_step2(a_pos, c_pos, target_position, a_g)
    changed_a_rot_mat = changed_a_rot_mat @ local_a_rot_diff_mat

    return changed_a_rot_mat, changed_b_rot_mat


class Jacobian_IK:
    def jacobian_IK(self, desired_position, original_rotmats, original_positions, offsets,
                    threshold=0.001, max_iter=800, beta=0.05):
        rotmat_list = copy.deepcopy(original_rotmats)
        position_list = np.array(original_positions)
        iteration = 0
        dist = utility.l2norm(desired_position-position_list[-1])
        radians = self.get_radians(rotmat_list)
        print("start_radians:", radians)

        # global_rot_axises, global_joint_rotmats = self.get_global_rot_info(rotmat_list)

        while (abs(dist) > threshold) and (iteration < max_iter):
            global_rot_axises, global_joint_rotmats = self.get_global_rot_info(rotmat_list)
            jacobian = self.calculate_jacobian(global_rot_axises, position_list)
            jacobian_t = jacobian.T
            position_delta = beta * (desired_position - position_list[-1])
            radians_delta = jacobian_t @ position_delta

            radians = radians + radians_delta

            rotmat_list = self.update_rotmat(radians_delta, global_rot_axises, global_joint_rotmats, rotmat_list)
            # rotmat_list = self.update_rotmat_local(radians, rotmat_list)

            position_list = self.update_position(rotmat_list, offsets, position_list)

            # global_rot_axises = self.update_global_axis(global_rot_axises, rotmat_list)

            dist = utility.l2norm(desired_position-position_list[-1])
            iteration += 1
        print("radians:", radians)
        print("desired:", desired_position, " | start_position:", original_positions[-1])
        print("arrived:", position_list[-1], "iteration: ", iteration, "dist: ", dist)
        return rotmat_list

    def update_global_axis(self, global_rot_axises, rotmat_list):
        new_global_rot_axises = np.array([])
        for index in range(len(global_rot_axises)):
            global_rot_mat = np.identity(3)
            for local_index in range(len(rotmat_list[:index+1])):
                global_rot_mat = global_rot_mat @ rotmat_list[local_index]
            new_global_rot_axis = utility.normalized(global_rot_mat @ global_rot_axises[index])
            new_global_rot_axis = np.reshape(new_global_rot_axis, (1, -1))
            new_global_rot_axises = utility.numpy_append(new_global_rot_axises, new_global_rot_axis)
        return new_global_rot_axises

    def get_radians(self, rotmat_list):
        radians = np.array([])
        for rotmat in rotmat_list:
            _, radian = utility.get_rot_axis_from(rotmat)
            radian = np.reshape(radian, (-1))
            radians = utility.numpy_append(radians, radian)
        return radians

    def get_global_rot_info(self, rotmat_list):
        global_rot_axises = np.array([])
        global_joint_rotmats = np.array([])
        for index in range(len(rotmat_list)):
            global_joint_rotmat = utility.dot_mat_sequence(rotmat_list, index)
            global_rot_axis, _ = utility.get_rot_axis_from(global_joint_rotmat)
            global_rot_axis = np.reshape(global_rot_axis, (1, -1))
            global_rot_axises = utility.numpy_append(global_rot_axises, global_rot_axis, axis=0)
            global_joint_rotmat = np.reshape(global_joint_rotmat, (1, 3, 3))
            global_joint_rotmats = utility.numpy_append(global_joint_rotmats, global_joint_rotmat, axis=0)
        return global_rot_axises, global_joint_rotmats

    def calculate_jacobian(self, global_rot_axises, position_list):
        end_effector = position_list[-1]
        jacobian_matrix = np.array([])
        for index in range(len(position_list)):
            global_rot_axis = global_rot_axises[index]
            joint_diff = end_effector - position_list[index]
            jacobi = np.cross(global_rot_axis, joint_diff)
            jacobi = np.reshape(jacobi, (-1, 1))
            jacobian_matrix = utility.numpy_append(jacobian_matrix, jacobi, axis=1)
        return jacobian_matrix

    def calculate_jacobian_local(self, local_rotmats, position_list):
        end_effector = position_list[-1]
        jacobian_matrix = np.array([])
        for index in range(len(position_list)):
            local_rot_axis = utility.normalized(utility.log(local_rotmats[index]))
            joint_diff = end_effector - position_list[index]
            jacobi = np.cross(local_rot_axis, joint_diff)
            jacobi = np.reshape(jacobi, (-1, 1))
            jacobian_matrix = utility.numpy_append(jacobian_matrix, jacobi, axis=1)
        return jacobian_matrix

    def update_rotmat(self, radians_delta, global_rot_axises, global_rotmats, rotmats):
        new_rot_mats = np.array([])
        for index in range(len(global_rot_axises)):
            if index >= 2:
                radian_delta = radians_delta[index]
            else:
                radian_delta = 0
            radian_delta = radians_delta[index]
            global_rot_axis = global_rot_axises[index]
            local_rot_axis = utility.normalized(global_rotmats[index].T @ global_rot_axis)
            local_rot_mat_diff = utility.exp(radian_delta * local_rot_axis)
            rotmat = rotmats[index]
            new_local_rot_mat = rotmat @ local_rot_mat_diff
            new_local_rot_mat = np.reshape(new_local_rot_mat, (1, 3, 3))
            new_rot_mats = utility.numpy_append(new_rot_mats, new_local_rot_mat, axis=0)
        return new_rot_mats

    def update_rotmat_local(self, radians, rotmats):
        new_rot_mats = np.array([])
        for index in range(len(radians)):
            local_rot_axis = utility.normalized(utility.log(rotmats[index]))
            new_local_rot_mat = utility.exp(radians[index] * local_rot_axis)
            new_local_rot_mat = np.reshape(new_local_rot_mat, (1, 3, 3))
            new_rot_mats = utility.numpy_append(new_rot_mats, new_local_rot_mat, axis=0)
        return new_rot_mats

    def update_position(self, rotmat_list, offset_list, pos_list):
        new_global_poses = np.array([])
        for index in range(len(rotmat_list)):
            global_pos = FK(rotmat_list[:index+1], offset_list[:index+1], pos_list[0])
            global_pos = np.reshape(global_pos, (1, -1))
            new_global_poses = utility.numpy_append(new_global_poses, global_pos, axis=0)
        return new_global_poses


class Jacobian_IK_3axises:

    def jacobian_IK(self, desired_position, original_rotmats, original_positions, offsets,
                    threshold=0.001, max_iter=800, beta=0.05):
        rotmat_list = copy.deepcopy(original_rotmats)
        position_list = np.array(original_positions)
        iteration = 0
        dist = utility.l2norm(desired_position - position_list[-1])
        while (abs(dist) > threshold) and (iteration < max_iter):
            global_joint_rotmats = self.get_global_rot_info(rotmat_list)
            jacobian_x, jacobian_y, jacobian_z = self.calculate_jacobian(position_list)
            jacobian_x_t = jacobian_x.T
            jacobian_y_t = jacobian_y.T
            jacobian_z_t = jacobian_z.T
            position_delta = beta * (desired_position - position_list[-1])
            radians_x_delta = jacobian_x_t @ position_delta
            radians_y_delta = jacobian_y_t @ position_delta
            radians_z_delta = jacobian_z_t @ position_delta

            rotmat_list = self.update_rotmat(radians_x_delta, radians_y_delta, radians_z_delta,
                                             global_joint_rotmats, rotmat_list)

            position_list = self.update_position(rotmat_list, offsets, position_list)
            dist = utility.l2norm(desired_position - position_list[-1])
            iteration += 1
        # print("desired:", desired_position, " | start_position:", original_positions[-1])
        # print("arrived:", position_list[-1], "iteration: ", iteration, "dist: ", dist)
        return rotmat_list

    def get_global_rot_info(self, rotmat_list):
        global_joint_rotmats = np.array([])
        for index in range(len(rotmat_list)):
            global_joint_rotmat = utility.dot_mat_sequence(rotmat_list, index)
            global_joint_rotmat = np.reshape(global_joint_rotmat, (1, 3, 3))
            global_joint_rotmats = utility.numpy_append(global_joint_rotmats, global_joint_rotmat, axis=0)
        return global_joint_rotmats

    def calculate_jacobian(self, position_list):
        end_effector = position_list[-1]
        jacobian_x_matrix = np.array([])
        jacobian_y_matrix = np.array([])
        jacobian_z_matrix = np.array([])
        global_rot_x_axis = np.array([1., 0., 0.])
        global_rot_y_axis = np.array([0., 1., 0.])
        global_rot_z_axis = np.array([0., 0., 1.])
        for index in range(len(position_list)):
            joint_diff = end_effector - position_list[index]
            jacobi_x = np.cross(global_rot_x_axis, joint_diff)
            jacobi_x = np.reshape(jacobi_x, (-1, 1))
            jacobian_x_matrix = utility.numpy_append(jacobian_x_matrix, jacobi_x, axis=1)
            jacobi_y = np.cross(global_rot_y_axis, joint_diff)
            jacobi_y = np.reshape(jacobi_y, (-1, 1))
            jacobian_y_matrix = utility.numpy_append(jacobian_y_matrix, jacobi_y, axis=1)
            jacobi_z = np.cross(global_rot_z_axis, joint_diff)
            jacobi_z = np.reshape(jacobi_z, (-1, 1))
            jacobian_z_matrix = utility.numpy_append(jacobian_z_matrix, jacobi_z, axis=1)
        return jacobian_x_matrix, jacobian_y_matrix, jacobian_z_matrix

    def update_rotmat(self, radians_x_delta, radians_y_delta, radians_z_delta,
                      global_rotmats, rotmats):
        new_rot_mats = np.array([])
        joint_num = len(global_rotmats)
        total_square = 0
        for index in range(joint_num):
            total_square = total_square + index*index
        for index in range(joint_num):
            delta_x = (index*index/total_square) * radians_x_delta[index]
            delta_y = ((index+1)*(index+1)/total_square) * radians_y_delta[index]
            delta_z = (index*index/total_square) * radians_z_delta[index]
            temp_rot_vec = np.array([delta_x, delta_y, delta_z])
            temp_rot_axis = utility.normalized(temp_rot_vec)
            temp_rot_radian = utility.l2norm(temp_rot_vec)
            local_rot_axis = utility.normalized(global_rotmats[index].T @ temp_rot_axis)
            local_rot_mat_diff = utility.exp(temp_rot_radian * local_rot_axis)
            rotmat = rotmats[index]
            new_local_rot_mat = rotmat @ local_rot_mat_diff
            new_local_rot_mat = np.reshape(new_local_rot_mat, (1, 3, 3))
            new_rot_mats = utility.numpy_append(new_rot_mats, new_local_rot_mat, axis=0)
        return new_rot_mats

    def update_position(self, rotmat_list, offset_list, pos_list):
        new_global_poses = np.array([])
        for index in range(len(rotmat_list)):
            global_pos = FK(rotmat_list[:index+1], offset_list[:index+1], pos_list[0])
            global_pos = np.reshape(global_pos, (1, -1))
            new_global_poses = utility.numpy_append(new_global_poses, global_pos, axis=0)
        return new_global_poses


def FK(rot_list, offset_list, root_position):
    transform = np.identity(4, dtype=float)
    my_position = offset_list[-1]
    my_position_affine = np.array([0., 0., 0., 1.])
    my_position_affine[:3] = my_position
    for index in range(len(rot_list)-1):
        before_rot_affine = utility.make_affine(transform=rot_list[index], affine_type="rotation")
        before_trans_affine = utility.make_affine(translation=offset_list[index], affine_type="translation")
        transform = transform @ before_trans_affine @ before_rot_affine
    root_trans_affine = utility.make_affine(translation=root_position, affine_type="translation")
    transform = root_trans_affine @ transform

    pos_affine = transform @ my_position_affine
    global_pos = pos_affine[:3]

    return global_pos


def main():
    ...

if __name__ == '__main__':
    main()
