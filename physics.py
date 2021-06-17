import numpy as np
import math

import utility
import dynamics_class


def gravity(y, data, g_coeff, g_resist_coeff, g_orientation=np.array([0., -1., 0.])):
    mass = data.get_mass()
    g_constant = g_coeff
    g_resist_constant = g_resist_coeff
    data_dim = data.get_dimension()
    velocity = y[data_dim:]
    g_velocity = np.array([0., velocity[1], 0.])

    # gravity resist version
    force = mass * g_constant * g_orientation - g_resist_constant * g_velocity

    # simple version
    # force = mass * g_constant * g_orientation

    return force


def viscous_drag(y, data):
    data_dim = data.get_dimension()
    velocity = y[data_dim:]
    drag_coeff = data.get_coeff_dict()["viscous_drag"]

    force = -drag_coeff*velocity

    return force


def spring(y, data):
    data_dim = data.get_dimension()
    position = y[:data_dim]
    velocity = y[data_dim:]
    connections = data.get_spring_dict()
    connected_particles = connections.keys()

    spring_force = np.array([0., 0., 0.])

    for particle in connected_particles:
        spring = connections[particle]
        y2 = particle.get_state()
        position2 = y2[:data_dim]
        velocity2 = y2[data_dim:]
        damped_coeff = spring.get_damped_coeff()
        spring_coeff = spring.get_spring_coeff()
        rest_len = spring.get_rest_len()

        delta_x = position - position2
        delta_v = velocity - velocity2
        coeff = -(spring_coeff * (utility.l2norm(delta_x) - rest_len) +
                  damped_coeff * (np.dot(delta_v, utility.normalized(delta_x))))
        force = coeff * utility.normalized(delta_x)
        spring_force += force

    return spring_force


def friction(contact_plane_info, total_force, y, data, threshold=0.001):
    dim = data.get_dimension()
    velocity = y[dim:]
    cor_normal = contact_plane_info[1]
    friction_particle = contact_plane_info[0]
    static_friction_coeff = friction_particle.get_coeff_dict()["static_friction"]
    kinetic_friction_coeff = friction_particle.get_coeff_dict()["kinetic_friction"]

    normal_force, tangential_force = decompose_vector(total_force, cor_normal)
    if np.dot(utility.normalized(normal_force), cor_normal) > 0:
        plane_normal_force = np.array([0., 0., 0.])
    else:
        plane_normal_force = -normal_force
    normal_velocity, tangential_velocity = decompose_vector(velocity, cor_normal)

    static_friction_mag = -static_friction_coeff * utility.l2norm(plane_normal_force)
    kinetic_friction_mag = -kinetic_friction_coeff * utility.l2norm(plane_normal_force)
    kinetic_friction = kinetic_friction_mag * utility.normalized(tangential_velocity)

    # todo: 원인 - friction force * time delta 값이 velocity보다 살짝 커서 아주 작은 -값이 되고 다시 -값에 대한 delta v가
    # todo        더해져서 +값이 되는게 반복됨
    if utility.l2norm(tangential_velocity) > 0.:
        print("kinetic")
        print("total_force: ", total_force, "total_vel:", velocity,
              "tangential_velocity: ", tangential_velocity, "tangential_force: ", kinetic_friction)
        friction_force = kinetic_friction
    else:
        print("static")
        if utility.l2norm(tangential_force) <= abs(static_friction_mag):
            friction_force = -tangential_force
        else:
            friction_force = static_friction_mag * utility.normalized(tangential_force)
        print("total_force: ", total_force, "total_vel:", velocity,
              "tangential_velocity: ", tangential_velocity, "tangential_force: ", friction_force)
    return friction_force


def decompose_vector(vector, opponent_normal):
    normal_projected_scale = np.dot(vector, opponent_normal)
    normal_vector = opponent_normal * normal_projected_scale
    tangential_vector = vector - normal_vector
    return normal_vector, tangential_vector


def take_new_state(keys, step_funcs, funcs, time_step, particles, collision_epsilon=0.01, contact_epsilon=0.01):
    # 0. 초기화
    t_curs = {}
    y_curs = {}
    new_states = {}
    new_times = {}
    derivs = {}
    time_delta = time_step
    for key in keys:
        t_curs.update({key: particles[key].get_time()})
        y_curs.update({key: particles[key].get_state()})
        particles[key].clear_collisions()
    print("time: ", t_curs)
    print("state: ", y_curs)

    # 1. 일단 계산
    # contact 여부 확인
    for key in keys:
        y_cur = y_curs[key]
        t_cur = t_curs[key]
        particle = particles[key]
        step_func = step_funcs[key]
        func = funcs[key]
        contact_detection(y_cur, particle, contact_epsilon=contact_epsilon)
        new_state, new_time, deriv = step_func(func, t_cur, y_cur, time_delta, particle)

        new_states.update({key: new_state})
        new_times.update({key: new_time})
        derivs.update({key: deriv})

    print("semi-state: ", new_states)

    # 2. 충돌 check
    for key in keys:
        new_state = new_states[key]
        particle = particles[key]
        collision_detection(new_state, particle, collision_epsilon=collision_epsilon)
    # 3. 충돌 시점 t 정보 구하기 (info = [충돌이 일어난 t delta, [충돌이 발생한 particle 쌍 list들, normal]])
    collision_time_info = get_first_collision_info(keys, particles, collision_epsilon, time_step)
    # 4. 충돌 시점에 충돌 발생한 particle 속성 (속도, 위치) 갱신
    while len(collision_time_info) > 0:
        print(collision_time_info)
        collision_time_delta = collision_time_info[0]
        collision_particle_pair_lists = collision_time_info[1]
        for collision_pair_info in collision_particle_pair_lists:
            collision_obj = collision_pair_info[0]
            collision_sub = collision_pair_info[1]
            for key in keys:
                particle = particles[key]
                if particle == collision_obj:
                    y_cur = y_curs[key]
                    t_cur = t_curs[key]
                    deriv = derivs[key]
                    step_func = step_funcs[key]
                    func = funcs[key]
                    collision_normal = collision_pair_info[2]

                    t_new = t_cur + collision_time_delta
                    y_new = y_cur + deriv * collision_time_delta
                    last_time_delta = time_delta - collision_time_delta

                    y_new = collision_response(y_new, collision_normal, particle, collision_sub)
                    print("response:", y_new)
                    contact_detection(y_new, particle, contact_epsilon=contact_epsilon)
                    new_state, new_time, deriv = step_func(func, t_new, y_new, last_time_delta, particle)
                    new_states.update({key: new_state})
                    new_times.update({key: new_time})
                    derivs.update({key: deriv})

        for key in keys:
            new_state = new_states[key]
            particle = particles[key]
            collision_detection(new_state, particle, collision_epsilon=collision_epsilon)
        collision_time_info = get_first_collision_info(keys, particles, collision_epsilon, time_step)

    return new_states, new_times


def euler_step(func, t_cur, y_cur, time_delta, data):
    data.clear_force()
    deriv = func(t_cur, y_cur, data)
    delta = deriv * time_delta
    y_new = y_cur + delta
    t_new = t_cur + time_delta
    return y_new, t_new, deriv


def mid_point_step(func, t_cur, y_cur, time_delta, data):
    data.clear_force()
    deriv = func(t_cur, y_cur, data)
    delta = deriv * time_delta
    t_new = t_cur + time_delta/2
    y_new = y_cur + delta/2

    data.clear_force()
    mid_point_deriv = func(t_new, y_new, data)
    mid_point_delta = mid_point_deriv * time_delta
    y_new = y_cur + mid_point_delta
    t_new = t_cur + time_delta
    return y_new, t_new, deriv


# TODO: 임시로 바닥과만 충돌 체크 --> Particle간, Particle과 plane 간 충돌 체크 필요
def collision_detection(state, particle, opponent_particle=None, collision_epsilon=0.01):
    # todo:임시로 바닥 생성
    plane1 = dynamics_class.Particle(init_position=np.array([0., 0., 1.]))
    plane2 = dynamics_class.Particle(init_position=np.array([1., 0., 0.]))
    plane3 = dynamics_class.Particle(init_position=np.array([0., 0., -1.]))
    plane4 = dynamics_class.Particle(init_position=np.array([-1., 0., 0.]))
    plane_particle = dynamics_class.PlaneCOMParticle(plane1, plane2, plane3, plane4)

    particle.clear_collisions()

    # todo: 다른 particle들과 collision detect ==> normal_particle(plane COM particle)도 이용
    # ===========================================
    opponent_state = plane_particle.get_state()
    opponent_data = plane_particle
    is_collision, _, opponent_normal = check_collision_contact(state, particle, opponent_state, opponent_data,
                                                               collision_epsilon=collision_epsilon)
    if is_collision:
        print("collision")
        collision_particle = opponent_data
        particle.update_collision(collision_particle, opponent_normal)
    # ===========================================


# TODO: 임시로 바닥과만 접촉 체크 --> Particle간, Particle과 plane 간 충돌 체크 필요
def contact_detection(state, particle, opponent_particle=None, contact_epsilon=0.01):
    # todo:임시로 바닥 생성
    plane1 = dynamics_class.Particle(init_position=np.array([0., 0., 1.]))
    plane2 = dynamics_class.Particle(init_position=np.array([1., 0., 0.]))
    plane3 = dynamics_class.Particle(init_position=np.array([0., 0., -1.]))
    plane4 = dynamics_class.Particle(init_position=np.array([-1., 0., 0.]))
    plane_particle = dynamics_class.PlaneCOMParticle(plane1, plane2, plane3, plane4)

    particle.clear_contacts()

    # todo: 다른 particle들과 contact detect ==> normal_particle(plane COM particle)도 이용
    # ===========================================
    opponent_state = plane_particle.get_state()
    opponent_data = plane_particle
    _, is_contact, opponent_normal = check_collision_contact(state, particle, opponent_state, opponent_data,
                                                             contact_epsilon=contact_epsilon)
    if is_contact:
        print("contact")
        contact = opponent_data
        particle.update_contact(contact, opponent_normal)
    # ===========================================


def check_collision_contact(state, particle, opponent_state, opponent_data,
                            collision_epsilon=0.01, contact_epsilon=0.01):
    is_collision = False
    is_contact = False

    dim = particle.get_dimension()
    particle_pos = state[:dim]
    particle_vel = state[dim:]
    opponent_pos = opponent_state[:dim]

    opponent_normal = opponent_data.get_normal(particle_pos)
    diff_vec = particle_pos - opponent_pos

    collision_check_dist = np.dot(diff_vec, opponent_normal)
    normalized_vel = utility.normalized(particle_vel)
    orientation_check = np.dot(opponent_normal, normalized_vel)

    if collision_check_dist <= collision_epsilon and orientation_check < 0.:
        is_collision = True
    elif abs(collision_check_dist) <= collision_epsilon and abs(orientation_check) < contact_epsilon:
        is_contact = True

    return is_collision, is_contact, opponent_normal


def get_first_collision_info(keys, particles, collision_epsilon, time_step):
    collision_check_epsilon = 0.00003
    collision_info = []
    collision_objects = []
    first_collision_time_delta = time_step
    for key in keys:
        particle = particles[key]
        collisions = particle.get_collisions()

        for collision in collisions:
            collision_particle = collision[0]
            collision_particle_normal = collision[1]
            dim = particle.get_dimension()
            state = particle.get_state()
            pos = state[:dim]
            vel = state[dim:]
            opponent_state = collision_particle.get_state()
            opponent_pos = opponent_state[:dim]
            opponent_vel = opponent_state[dim:]
            normal_pos_diff_mag = np.dot(pos - opponent_pos, collision_particle_normal)
            normal_vel_diff_mag = np.dot(vel - opponent_vel, collision_particle_normal)

            target_vel = collision_epsilon - normal_pos_diff_mag
            collision_time_delta = target_vel / normal_vel_diff_mag

            if collision_time_delta < first_collision_time_delta:
                first_collision_time_delta = collision_time_delta
                collision_objects = list()
                collision_objects.append([particle, collision_particle, collision_particle_normal])
                collision_info = [first_collision_time_delta, collision_objects]
            elif abs(collision_time_delta - first_collision_time_delta) <= collision_check_epsilon:
                collision_objects.append([particle, collision_particle, collision_particle_normal])
                collision_info[1] = collision_objects
    return collision_info


# TODO: 임시로 바닥과만 튕겨짐 --> 움직이는 particle 및 plane 간, 고정된 particle 및 plane 과의 충돌 처리 필요
# TODO: 충돌 직전 상태로 되돌리는 기능 만들기
def collision_response(y, collision_normal, particle, oppo_particle, vel_threshold=0.04):
    dim = particle.get_dimension()
    position = y[:dim]
    velocity = y[dim:]

    final_velocity = np.array([0., 0., 0.])
    elasticity_coeff = oppo_particle.get_coeff_dict()["elasticity"]
    normal_projected_scale = np.dot(velocity, collision_normal)
    normal_projected_vel = collision_normal * normal_projected_scale
    tangent_projected_vel = velocity - normal_projected_vel
    normal_projected_coll_vel = elasticity_coeff * normal_projected_vel
    coll_vel = tangent_projected_vel - normal_projected_coll_vel
    if utility.l2norm(normal_projected_coll_vel) < vel_threshold:
        coll_vel = tangent_projected_vel
    final_velocity = final_velocity + coll_vel
    final_state = np.concatenate((position, final_velocity), axis=None)

    return final_state


def deriv_eval(t, y, data):
    total_force = np.array([0., 0., 0.])
    coeff_dict = data.get_coeff_dict()
    g_coeff = coeff_dict["g"]
    g_resist_coeff = coeff_dict["g_resist"]
    g_force = gravity(y, data, g_coeff, g_resist_coeff)
    total_force += g_force
    spring_force = spring(y, data)
    total_force += spring_force
    data.accumulate_force(total_force)

    if len(data.get_contacts()) > 0:
        contacts = data.get_contacts()
        for contact in contacts:
            cor_normal = contact[1]
            normal_force, tangential_force = decompose_vector(total_force, cor_normal)
            if np.dot(normal_force, cor_normal) >= 0:
                plane_normal_force = np.array([0., 0., 0.])
            else:
                plane_normal_force = -normal_force
            friction_force = friction(contact, total_force, y, data)
            viscous_force = viscous_drag(y, data)
            data.accumulate_force(viscous_force)
            data.accumulate_force(friction_force)
            data.accumulate_force(plane_normal_force)
            print(data.get_force()/data.get_mass())
            print(y[3:])

    elif len(data.get_collisions()) > 0:
        collisions = data.get_collisions()
        for collision in collisions:
            cor_normal = collision[1]
            normal_force, tangential_force = decompose_vector(total_force, cor_normal)
            if np.dot(normal_force, cor_normal) >= 0:
                plane_normal_force = np.array([0., 0., 0.])
            else:
                plane_normal_force = -normal_force
            friction_force = friction(collision, total_force, y, data)
            viscous_force = viscous_drag(y, data)
            data.accumulate_force(viscous_force)
            data.accumulate_force(friction_force)
            data.accumulate_force(plane_normal_force)
            print(data.get_force()/data.get_mass())

    dot_v = data.get_force()/data.get_mass()
    dim = data.get_dimension()
    dot_x = y[dim:]

    derivs = np.concatenate((dot_x, dot_v), axis=None)
    return derivs
