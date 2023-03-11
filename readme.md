# Intro

This is a simulation written for my MECH 223 course. It simulates the behaviour of a ball rolling down a well that is defined in the project specifications.

The principle of the physical simulation is newtonian mechanics. The geometrical constraints is incorported with conservation energy, forced displacement in the normal direction, and forced tangency of the velocity vector.

# How to use this github

## How to setup

1. Install python.
2. `python -m pip install -r requirements.txt`

## How to run a simulation

To start a new simulation:

1. Put in desired configuration into the config.yaml file
2. Run `python -m simulation.py` and wait until the console log k value is greater than the desired number of trials.
3. Press `Ctrl + c` in console to terminate simulation.
4. Run `python -m visaulization.py` and
5. Press `Ctrl + c` in console to terminate visualization.

Alternatively, to get more data for a previous simulation. Replace step 1 with:
Copy the config file next to saved data into the config file in root directory.

## How to use config.yaml?

meta_property are properties that are not related to the physical setups

- save_folder_name identify different runs.
  If save_folder_name didn't change, all new data will be added on to existing data.
- time_step is the dt in the internal physics loop.
- x/y intial max/min defines the range of the possibility space to examine.
  To inspect a specific region of the configuration space, change the x y initial values without changing the save folder.

simulation_property are properties that are related to the physical setups

- ball_radius is the radius of the ball to be launched in meters.
- density is the density of the ball to be launched in kg/m^3^.
- rolling_friction_coefficient is a dimension less number regarding rolling friction.

## How to read the visualization?

The left plot is the physical trajectory of a few filtered out trajectories.

- It is used to examine whether the trajectory is reasonable.
- It is the least abstract data.

The middle plot is the launch configuration space.

- The horizontal axis represent the distance between the center of the well and the line formed by the initial velocity vector.
- The vertical axis represent the magnitude of the initial velocity vector.
- The color represent the performance of the launch configuration. It is not accurate, but the general shape is useful.
- The solid line represent the velocity of stable orbit with the said radius.

The right plot is the energy over time graph of the few filtered out trajectories.

- It is used to inspect whether the simulation has created free energe.

---

# Dev Side

## Todo List

- [ ] Vectorize.
- [ ] More accurate normal force calculation.

## Profiling

1. To run: `python -m cProfile -o simulation.profile simulation.py`
2. To see: `snakeviz simulation.profile`
