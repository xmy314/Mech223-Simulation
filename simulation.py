from math import sqrt, pi
import numpy as np
from random import random
from geometry import Well
import os
import yaml
from util import *

# region read configuration file
with open("config.yaml", "r") as yamlfile:
    config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)

# simulation_property
simulation_property = config_data["simulation_property"]
radius = simulation_property["ball_radius"]
density = simulation_property["density"]
rolling_friction_coefficient = simulation_property["rolling_friction_coefficient"]

# meta property
meta_property = config_data["meta_property"]
time_step = meta_property["time_step"]
x_initial_range = [meta_property["x_intial_min"], meta_property["x_intial_max"]]
y_initial_range = [meta_property["y_intial_min"], meta_property["y_intial_max"]]
save_folder_name = meta_property["save_folder_name"]

# endregion read configuration file

# pre computation
inertia = (4/3)*pi*(radius**3)*density
angular_inertia = (2/5)*(inertia*radius**2)
well = Well(radius)


class Condition:
    # assume perfect sphere always rolling without sliding
    def __init__(self, position, velocity, time_stamp=0):
        # velocity is half a time step before position as outlined by leap frog integration
        self.position = position
        self.velocity = velocity
        self.time = time_stamp
        self.fix_contact()

    def fix_contact(self):
        self.is_in_contact, self.contact_position = well.getContact(self.position)

        self.contact_distance = 0

        if not self.contact_position is None:
            self.is_in_contact = True
            self.contact_distance = np.linalg.norm(self.position-self.contact_position)

            self.unit_normal = normalize(self.position-self.contact_position)
            self.unit_binormal = np.cross(self.unit_normal, normalize(self.velocity))
            self.unit_velocity = np.cross(self.unit_binormal, self.unit_normal)

            # the following two are quick hacks to prevent glitching.
            # scarfices accuracy overall for more less glitch
            # use energy conservation to get a "possible" result
            energy = inertia*self.position[2]*9.81 + 0.5*inertia*(np.linalg.norm(self.velocity))**2

            # force position tangency
            self.position = self.position-(self.contact_distance-radius)*self.unit_normal

            remaining_energy = energy-inertia*self.position[2]*9.81
            if remaining_energy <= 0:
                remaining_energy = 0
            remaining_velocity = np.sqrt(2*remaining_energy/(inertia))

            # change_d = self.contact_distance-radius
            # change_v = np.linalg.norm(remaining_velocity*self.unit_velocity-self.velocity)/np.linalg.norm(self.velocity)
            # print(f"d: {change_d:.10f},v: {change_v:.10f}")

            # force velocity tangency
            self.velocity = remaining_velocity*self.unit_velocity

    def next(self):
        # forces
        force = np.array([0, 0, 0], dtype=np.float64)

        # weight
        force += inertia*np.array([0, 0, -9.81])

        # drag
        # force += 0.5*0.001293*(-self.velocity*abs(np.linalg.norm(self.velocity)))*0.47*pi*(radius**2)

        # normal
        if self.is_in_contact:
            desired_acc_out = well.getNormalAcceleration(self.contact_position,self.velocity)
            # desired_acc_out = 0
            desired_force_out = desired_acc_out*(inertia)
            force_out = np.dot(force, self.unit_normal)
            if force_out < desired_force_out:
                normal_force = (desired_force_out-force_out)*self.unit_normal
                # TODO: even though normal force is perpendicular to velocity, it only does no work for very small timesteps.
                proportion=time_step*np.linalg.norm(normal_force/inertia)/np.linalg.norm(self.velocity)
                print(f"{proportion:7.5f}")
                force+=normal_force

        # apply forces 
        acc = force/(inertia)

        self.position = self.position+self.velocity*time_step
        self.velocity = self.velocity+acc*time_step
        # self.position = self.position+self.velocity*time_step+0.5*acc*(time_step**2)

        self.time += time_step

        self.fix_contact()

    def isvalid(self):
        return (self.position[0] >= -1 and self.position[0] <= 1 and
                self.position[1] >= -1 and self.position[1] <= 1 and
                self.position[2] >= -1 and self.position[2] <= 1)


# prepare the save folder or read from prvious saves
save_folder_path = os.path.join("results", save_folder_name)

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    with open(os.path.join(save_folder_path, "config.yaml"), "w+") as yamlfile:
        yaml.dump(config_data, yamlfile)

    with open(os.path.join(save_folder_path, "progress.txt"), "w+") as fi:
        fi.write("0")

    k = 0
else:
    with open(os.path.join(save_folder_path, "progress.txt"), "r") as fi:
        k = int(fi.read())

if k > 0:
    with open(os.path.join(save_folder_path, "phase_save.npy"), 'rb') as f:
        phase_data = list(np.load(f))
else:
    phase_data = []

while True:
    # generate initial location
    initial_p_x = random()*(x_initial_range[1]-x_initial_range[0])+x_initial_range[0]
    if initial_p_x**2 > (0.45**2):
        continue
    initial_position = np.array([initial_p_x, -np.sqrt((0.45**2)-initial_p_x**2), radius], dtype=np.float64)

    # generate initial velocity
    initial_v_y = random()*(y_initial_range[1]-y_initial_range[0])+y_initial_range[0]
    initial_velocity = np.array([0, initial_v_y, 0], dtype=np.float64)

    if np.linalg.norm(initial_v_y) < 0.0001:
        continue

    initial_position = np.array([initial_p_x, -np.sqrt((0.45**2)-initial_p_x**2), radius], dtype=np.float64)

    initial_angular_velocity = np.array([0, 0, 0], dtype=np.float64)

    condition = Condition(initial_position, initial_velocity)
    data = []
    halt_counter = 0
    while True:
        speed = np.linalg.norm(condition.velocity)
        data.append([condition.time, *condition.position, speed, condition.contact_distance])
        if (not condition.isvalid()):
            break

        if (speed == 0):
            halt_counter += 1
        else:
            halt_counter = 0
        if halt_counter == 100:
            break

        if condition.time >= 20:
            break

        condition.next()
    print(f"k:{k:7}     t:{condition.time:7.3f}")

    data = np.array(data).T
    with open(os.path.join(save_folder_path, "sequential_save.npy"), 'ab+') as f:
        np.save(f, data)

    phase_data.append(np.array([initial_p_x, initial_v_y, data[0, -1]]))
    with open(os.path.join(save_folder_path, "phase_save.npy"), 'wb+') as f:
        np.save(f, np.array(phase_data))

    k += 1
    with open(os.path.join(save_folder_path, "progress.txt"), "w+") as fi:
        fi.write(str(k))
