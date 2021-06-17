import numpy as np
import copy

import physics
import utility

class Particle:
    def __init__(self, init_position=np.array([0., 0., 0.]), init_velocity=np.array([0., 0., 0.]), mass=1.):
        self.position = init_position
        self.velocity = init_velocity
        self.mass = mass
        self.force = np.array([0., 0., 0.])
        self.past_position = None
        self.past_velocity = None
        self.spring_dict = {}       # {Particle:Spring}
        self.coeff_dict = {"g": 9.8, "g_resist": 0., "viscous_drag": 0.,
                           "static_friction": 1.0, "kinetic_friction": 0.8,
                           "elasticity": 0.5}
        self.contacts = []      # [[particle, normal]]
        self.collisions = []
        self.is_fixed = False
        self.time = 0.

    def get_dimension(self):
        return np.shape(self.position)[0]

    def get_mass(self):
        return self.mass

    def get_normal(self, opponent_particle_pos):
        diff = opponent_particle_pos - self.position
        normal = utility.normalized(diff)
        return normal

    def set_position(self, position):
        self.position = position

    def set_velocity(self, velocity):
        self.velocity = velocity

    def update_time(self, time):
        self.time = time

    def get_time(self):
        return self.time

    def update_collision(self, collision, normal):
        self.collisions.append([collision, normal])

    def get_collisions(self):
        return self.collisions

    def get_contacts(self):
        return self.contacts

    def update_contact(self, contact, normal):
        self.contacts.append([contact, normal])

    def update_fixed(self, fixed):
        self.is_fixed = fixed

    def set_state(self, position, velocity):
        dim = self.get_dimension()
        if np.shape(position)[0] == dim:
            self.set_position(position)
            self.set_velocity(velocity)

    def get_state(self):
        state = np.concatenate((self.position, self.velocity), axis=None)
        return state

    def update_state(self, state):
        dim = self.get_dimension()
        position = state[:dim]
        velocity = state[dim:]
        self.past_position = copy.deepcopy(self.position)
        self.past_velocity = copy.deepcopy(self.velocity)
        self.position = position
        self.velocity = velocity

    def get_force(self):
        return self.force

    def clear_force(self):
        self.force = np.array([0., 0., 0.])

    def clear_contacts(self):
        self.contacts = []

    def clear_collisions(self):
        self.collisions = []

    def accumulate_force(self, force):
        self.force = self.force + force

    def get_coeff_dict(self):
        return self.coeff_dict

    def set_coeff(self, item, value):
        self.coeff_dict.update({item: value})

    def make_spring_connection(self, particle, spring):
        self.spring_dict.update({particle: spring})

    def get_spring_dict(self):
        return self.spring_dict

class PlaneCOMParticle(Particle):
    def __init__(self, *joint_particles):
        super().__init__()
        self.position = np.array([0., 0., 0.])
        self.normal = np.array([0., 0., 0.])
        self.mass = 0.
        self.joint_particles = joint_particles
        self.update_info()

    def get_normal(self, opponent_particle):
        return self.normal

    def update_info(self):
        total_mass = 0.
        for particle in self.joint_particles:
            mass = particle.get_mass()
            total_mass = total_mass + mass

        for particle in self.joint_particles:
            state = particle.get_state()
            dim = particle.get_dimension()
            mass = particle.get_mass()
            self.position = self.position + (mass / total_mass) * state[:dim]
        self.mass = total_mass / len(self.joint_particles)
        dim = self.joint_particles[0].get_dimension()
        vec_1 = self.joint_particles[0].get_state()[:dim] - self.joint_particles[1].get_state()[:dim]
        vec_2 = self.joint_particles[0].get_state()[:dim] - self.joint_particles[-1].get_state()[:dim]
        self.normal = utility.normalized(np.cross(vec_1, vec_2))


class Spring:
    def __init__(self, particle_1, particle_2):
        self.damped_coeff = 10
        self.spring_coeff = 300
        self.rest_length = 0.5
        self.connection = [particle_1, particle_2]

    def set_damped_coeff(self, coeff):
        self.damped_coeff = coeff

    def set_spring_coeff(self, coeff):
        self.spring_coeff = coeff

    def get_damped_coeff(self):
        return self.damped_coeff

    def get_spring_coeff(self):
        return self.spring_coeff

    def get_connection(self):
        return self.connection

    def get_rest_len(self):
        return self.rest_length
