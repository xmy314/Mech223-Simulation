import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from math import sqrt, pi
from geometry import Well
import os
import yaml
from util import *


def is_interesting(serie):
    t, x, y, z, speed, contact = serie
    if np.all(np.linalg.norm(np.array([x, y]), axis=0) > 0.445):
        return False

    if speed[0] < 0.5:
        return False

    if x[0] >= 0.445:
        return False

    if (t[-1] < 1):
        return False
    return True


def plot_well(plot_handle):
    r, theta = np.meshgrid(np.linspace(0.1237, 0.45, 10), np.linspace(0, 2*pi, 6*4+1))
    x, y, z = r*np.cos(theta), r*np.sin(theta), np.reshape(well.h(r.flatten()), r.shape)
    plot_handle.plot_wireframe(x, y, z, zorder=1, linewidths=0.5)

    plot_handle.set_xlim([-0.5, 0.5])
    plot_handle.set_ylim([-0.5, 0.5])
    plot_handle.set_zlim([-0.5, 0.5])

    plot_handle.set_xlabel("X")
    plot_handle.set_ylabel("Y")
    plot_handle.set_zlabel("Z")


def plot_stability(plot_handle):
    # contact point to center radius
    l_t = np.linspace(0.1, 0.45, 1000)

    # slope at the contact point
    fpt = well.h(l_t, 1)
    # smth related to the normal vector
    btm = 1/np.sqrt(1+(fpt)**2)
    # ball center to well center distance
    x1 = l_t-ball_radius*fpt*btm

    v_stable = np.sqrt(x1*9.81*fpt)
    plot_handle.plot(x1, v_stable)

    plot_handle.set_xlabel("Initial Position(m)")
    plot_handle.set_ylabel("Initial Velocity(m/s)")


def plot_sequential(plot_handle, sequential_data):
    for serie in sequential_data:
        t, x, y, z, speed, contact = serie
        plot_handle.plot(x, y, z)

    plot_handle.set_xlim([-0.5, 0.5])
    plot_handle.set_ylim([-0.5, 0.5])
    plot_handle.set_zlim([-0.5, 0.5])

    plot_handle.set_xlabel("X")
    plot_handle.set_ylabel("Y")
    plot_handle.set_zlabel("Z")


def plot_energy(plot_handle, sequential_data):
    for serie in sequential_data:
        t, x, y, z, speed, contact = serie
        g_energy = 9.81*(z+0.2)
        k_energy = 0.5*speed**2
        plot_handle.plot(t, g_energy+k_energy)

    plot_handle.set_xlabel("Time")
    plot_handle.set_ylabel("Energy")


def plot_energy_loss(plot_handle, sequential_data):
    for serie in sequential_data:
        t, x, y, z, speed, contact = serie
        g_energy = 9.81*(z+0.2)
        k_energy = 0.5*speed**2
        energy = g_energy+k_energy
        plot_handle.plot(speed[:-1], change_convolve(energy))

    plot_handle.set_xlabel("speed")
    plot_handle.set_ylabel("Energy Loss")


def plot_configuration_space(plot_handle, configuration_space_data):
    px, vy, t = configuration_space_data
    cs = plot_handle.tricontourf(px, vy, t,
                                 locator=ticker.MaxNLocator(30),
                                 cmap="cool")
    _ = plt.colorbar(cs)

    plot_handle.set_xlabel("Initial Position(m)")
    plot_handle.set_ylabel("Initial Velocity(m/s)")


def plot_contact(plot_handle, sequential_data):
    for serie in sequential_data:
        t, x, y, z, speed, contact = serie
        plot_handle.plot(t, rollavg_convolve(contact, 21))

    plot_handle.set_xlabel("Time")
    plot_handle.set_ylabel("Percent time in contact")


def plot_speed(plot_handle, sequential_data):
    for serie in sequential_data:
        t, x, y, z, speed, contact = serie
        plot_handle.plot(t, speed)

    plot_handle.set_xlabel("Time")
    plot_handle.set_ylabel("Speed")


# region read saved data
with open("config.yaml", "r") as yamlfile:
    config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)

simulation_name = config_data["meta_property"]["save_folder_name"]
save_folder_path = os.path.join("results", simulation_name)

with open(os.path.join(save_folder_path, "progress.txt"), "r") as fi:
    N = int(fi.read())

ball_radius = config_data["simulation_property"]["ball_radius"]
well = Well(ball_radius)

with open(os.path.join(save_folder_path, "sequential_save.npy"), 'rb') as f:
    sequential_data = []
    skip = round(1/30/config_data["meta_property"]["time_step"])
    for i in range(N):
        serie = np.load(f)

        if (not is_interesting(serie)):
            continue

        serie = serie[:, ::skip]
        sequential_data.append(serie)

with open(os.path.join(save_folder_path, "phase_save.npy"), 'rb') as f:
    configuration_space_data = (np.load(f)).T

# endregion

# # the following region is a very comprehensive plot, but not useful for a big picture
# fig = plt.figure(figsize=(18, 5))
# seqential_graph = fig.add_subplot(1, 3, 1, projection='3d')
# phase_graph = fig.add_subplot(1, 3, 2)
# energy_graph = fig.add_subplot(1, 3, 3)

# plot_well(seqential_graph)
# plot_sequential(seqential_graph,sequential_data)
# plot_stability(phase_graph)
# plot_configuration_space(phase_graph,configuration_space_data)
# plot_energy_loss(energy_graph,sequential_data)

# fig.tight_layout()
# plt.savefig(os.path.join("debug",simulation_name+".png"))
# plt.show()

fig = plt.figure(figsize=(8, 6))
phase_graph = fig.add_subplot(1, 1, 1)

plot_stability(phase_graph)
plot_configuration_space(phase_graph, configuration_space_data)

plt.savefig(os.path.join("debug", simulation_name+".png"))
plt.show()
