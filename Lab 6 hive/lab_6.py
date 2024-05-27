from random import random, randint
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import time


class Particle:
    def __init__(self, init_cord_x_y, init_velocity_x_y, parameters, boundaries) -> None:
        self.cord_x_y = init_cord_x_y
        self.velocity_x_y = init_velocity_x_y
        self.W = parameters[0]
        self.C1 = parameters[1]
        self.C2 = parameters[2]
        self.local_best = None
        self.local_best_x_y = None
        self.global_best_x_y = None
        self.boundaries = boundaries

    def move(self):
        for i, velocity in enumerate(self.velocity_x_y):
            cord = self.cord_x_y[i]
            local_best = self.local_best_x_y[i]
            global_best = self.global_best_x_y[i]
            self.velocity_x_y[i] = self._calculate_velocity(velocity, cord, local_best, global_best)
        for i, cord in enumerate(self.cord_x_y):
            velocity = self.velocity_x_y[i]
            boundary = self.boundaries[i]
            if abs(self.cord_x_y[i] + velocity) >= boundary:
                self.velocity_x_y[i] = -self.velocity_x_y[i]
                velocity = self.velocity_x_y[i]
            self.cord_x_y[i] = self.cord_x_y[i] + velocity

    def _calculate_velocity(self, last_velocity, current_cord, local_best, global_best):
        local_impact = self.C1*random.uniform(0, 1)*(local_best-current_cord)
        global_impact = self.C2*random.uniform(0, 1)*(global_best-current_cord)
        inertia = last_velocity*self.W
        new_velocity = inertia + local_impact + global_impact
        return new_velocity

    def update_if_best_local_score(self, score):
        if self.local_best is None:
            self.local_best = score
            self.local_best_x_y = self.cord_x_y
        elif score < self.local_best:
            self.local_best = score
            self.local_best_x_y = self.cord_x_y
    
    def update_global_best(self, new_best_x_y):
        self.global_best_x_y = new_best_x_y.copy()

    def get_cords_x_y(self):
        return self.cord_x_y.copy()

    def modify_parameters(self, new_parameters):
        self.W = new_parameters[0]
        self.C1 = new_parameters[1]
        self.C2 = new_parameters[2]

class ParticleSwarmOptymalizer:
    def __init__(self, function, first_parameters, second_parameters, threshold) -> None:
        self.function = function
        self.first_parameters = first_parameters
        self.second_parameters = second_parameters
        self.threshold = threshold
        self.particles = None
        self.global_best = None
        self.first_iter = True
        self.fig, self.ax = plt.subplots()
        self.scat = None
        self.i = 0

    def create_swarm(self, particle_number):
        particles = []
        parameters = self.first_parameters
        boundaries = [4.5, 4.5]
        for _ in range(particle_number):
            random_start_x_y_cords = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5)]
            random_start_velocity = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
            particle = Particle(random_start_x_y_cords, random_start_velocity, parameters, boundaries)
            particles.append(particle)
        return particles

    def calculate_scores(self, particles, function):
        scores = []
        for particle in particles:
            cords_x_y = particle.get_cords_x_y()
            score = function(cords_x_y[0], cords_x_y[1])
            particle.update_if_best_local_score(score)
            scores.append(score)

        if self.global_best is None:
            self.global_best = min(scores)
            index_of_min_val = scores.index(min(scores))
            best_particle = particles[index_of_min_val]
            best_x_y = best_particle.get_cords_x_y()
            for particle in particles:
                particle.update_global_best(best_x_y)
        elif min(scores) < self.global_best:
            self.global_best = min(scores)
            index_of_min_val = scores.index(min(scores))
            best_particle = particles[index_of_min_val]
            best_x_y = best_particle.get_cords_x_y()
            for particle in particles:
                particle.update_global_best(best_x_y)

    def perform_optymalisation_test(self, frame):
        self.i += 1
        if self.i == self.threshold:
            for particle in self.particles:
                particle.modify_parameters(self.second_parameters)
        if self.first_iter == True:
            self.particles = self.create_swarm(1000)
            self.first_iter = False
            self.ax.set_xlim(-4.5, 4.5)
            self.ax.set_ylim(-4.5, 4.5)
            x = []
            y = []
            for particle in self.particles:
                x_y = particle.get_cords_x_y()
                x.append(x_y[0])
                y.append(x_y[1])
            self.scat = self.ax.scatter(x, y)
        self.calculate_scores(self.particles, self.function)
        for particle in self.particles:
            particle.move()
        x = []
        y = []
        for particle in self.particles:
            x_y = particle.get_cords_x_y()
            x.append(x_y[0])
            y.append(x_y[1])
        self.scat.set_offsets(np.c_[x, y])
        return self.scat,

    def perform_optymalisation_with_visualisation(self, max_iter):
        ani = animation.FuncAnimation(self.fig, self.perform_optymalisation_test,frames=max_iter, interval=100, blit=True, repeat=False)
        plt.show()

def function_to_optymize(x, y):
    res = pow((1.5-x-x*y), 2) + pow((2.25-x+pow(x*y, 2)), 2) + pow((2.625-x+pow(x*y, 3)), 2)
    return res

particle_swarm = ParticleSwarmOptymalizer(function_to_optymize, [1, 0.05, 0.05], [0.5, 0.5, 0.5], 100)
particle_swarm.perform_optymalisation_with_visualisation(200)
particles = particle_swarm.particles
print(particle_swarm.global_best, particles[0].global_best_x_y)


#first_params   second_params   threshold   max_iter    particle_number
#[1, 0.05, 0.05], [0.5, 0.5, 0.5], 100, 200, 1000          0.13178336458756887 [2.3758253174912594, -0.2557467347568813]
#[1, 0.05, 0.05], [0.5, 0.5, 0.5], 50,  100, 1000          0.13178336458756887 [2.3758253160645832, -0.2557467348811057]
#[1, 0.1, 0.1],   [0.5, 1.3, 1.3], 50,  100, 1000          0.1317833645875689  [2.3758253141392816, -0.2557467358982484]
#[1, 0.1, 0.1],   [0.5, 1.3, 1.3], 100, 200, 1000          0.13178336458756887 [2.375825318693521, -0.25574673512624513]
#[1, 0.05, 0.05], [0.5, 0.5, 0.5], 100, 200, 500           0.13178336458756884 [2.375825316922135, -0.2557467352859211]
#[1, 0.05, 0.05], [0.5, 0.5, 0.5], 50,  100, 500           0.13178336458756892 [2.3758253155782287, -0.255746736211649]
#[1, 0.1, 0.1],   [0.5, 1.3, 1.3], 50,  100, 500           0.13178336458756892 [2.375825316147881, -0.25574673419091215]
#[1, 0.1, 0.1],   [0.5, 1.3, 1.3], 100, 200, 500           0.13178336458756884 [2.375825317109716, -0.25574673531096814]
#[1, 0.05, 0.05], [0.5, 0.5, 0.5], 100, 200, 100           0.13178336458756884 [2.375825317166039, -0.25574673543120974]
#[1, 0.05, 0.05], [0.5, 0.5, 0.5], 50,  100, 100           0.13178336458756887 [2.3758253173091397, -0.25574673531794306]
#[1, 0.1, 0.1],   [0.5, 1.3, 1.3], 50,  100, 100           0.1317833645875799  [2.375825305267616, -0.25574675669743385]
#[1, 0.1, 0.1],   [0.5, 1.3, 1.3], 100, 200, 100           0.13178336458756887 [2.3758253174502793, -0.2557467347505872]

