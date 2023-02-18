from math import sqrt, pi
import numpy as np
from alive_progress import alive_bar
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os
import sympy

# material property
radius = 0.02
density = 1000
friction_coefficient = 0.3
rolling_friction_coefficient = 0.00001

restore_ratio = 0

# simulation property
time_step = 0.001
N = 10
x_intial_range = [3.00, 3.5]
y_intial_range = [0.5, 0.75]

# save property
save_folder = "results"

# pre computation
inertia = (4/3)*pi*(radius**3)*density
angular_inertia = (2/5)*(inertia*radius**2)


def normalize(vector):
    return vector/np.linalg.norm(vector)

def project(a,b):
    # project from a to b
    return (np.dot(a,b)/np.dot(b,b))*b

class Geometry:
    hole_origin = np.array([3.5, 1.9, 0])
    estimator = pickle.load(open("estimator.sav", 'rb'))
    tranformer = PolynomialFeatures(20)
    symbolic_r = sympy.symbols("r")
    #  -((r**2)-0.913*r+0.209)/(4.93*(r**2)+r+0.36)
    monster_func = (0.154/0.15134222365)*(-((symbolic_r**2)-(0.913240567396*symbolic_r)+0.208458255328)/(4.93*(symbolic_r**2)+symbolic_r+0.36))
    monster_func1 = sympy.diff(monster_func, symbolic_r)
    monster_func2 = sympy.diff(monster_func1, symbolic_r)

    @staticmethod
    def h(r, order=0):
        if order == 0:
            if r <= 0.15:
                return -0.457+2.02*r
            elif r <= 0.45:
                return float(Geometry.monster_func.subs(Geometry.symbolic_r, r))
            else:
                return 0
        elif order == 1:
            if r <= 0.15:
                return 2.02
            elif r <= 0.45:
                return float(Geometry.monster_func1.subs(Geometry.symbolic_r, r))
            else:
                return 0
        elif order == 2:
            if r <= 0.15:
                return 0
            elif r <= 0.45:
                return float(Geometry.monster_func2.subs(Geometry.symbolic_r, r))
            else:
                return 0

    @staticmethod
    def GetPosition(position_xy0):
        P_rel_O = (position_xy0-Geometry.hole_origin)
        P_rel_O[2] = 0
        height = Geometry.h(np.linalg.norm(P_rel_O))
        position_xyz = np.array([position_xy0[0], position_xy0[1], height])
        return position_xyz

    @staticmethod
    # numerically approximiate the point on surface closed to the ball
    # this function is an abomination
    def getContact(position):
        if (position[2] > radius):
            return [False, None]
        # first switch to rz

        P_rel_O = (position-Geometry.hole_origin)
        P_rel_O[2] = 0

        pos_rz = np.array([np.linalg.norm(P_rel_O), position[2]])
        P_rel_O = P_rel_O/np.linalg.norm(P_rel_O)

        # check regions
        if pos_rz[1] < -0.2-(radius):
            # definitely below the well
            return [False, None]
        elif pos_rz[1] < -0.49504950495*pos_rz[0]-0.13708570297:
            # almost below the well
            if (np.linalg.norm(pos_rz - np.array([0.1272, -0.2])) <= radius):
                ret = Geometry.hole_origin+P_rel_O * 0.1272+np.array([0, 0, -0.2])
                return [True, ret]
            else:
                return [False, None]
        elif pos_rz[1] < -0.49504950495*pos_rz[0]-0.0797425742574:
            # in the straight edge
            if pos_rz[1] < 2.02*pos_rz[0]-0.457+(radius)*2.25397426782:
                # is close enough
                contact_radius = (0.49504950495*pos_rz[0]+0.457+pos_rz[1])/(2.51504950495)
                ret = Geometry.hole_origin+P_rel_O*contact_radius + np.array([0, 0, Geometry.h(contact_radius)])
                return [True, ret]
            else:
                # is not close enough
                return [False, None]
        elif pos_rz[1] < -0.61735918881*pos_rz[0]-0.0613961216765:
            # between two wells
            if (np.linalg.norm(pos_rz - np.array([0.15, -0.154])) <= radius):
                ret = Geometry.hole_origin+P_rel_O * 0.15+np.array([0, 0, -0.154])
                return [True, ret]
            else:
                return [False, None]
        elif pos_rz[0] > 0.45:
            # out of the well region
            ret = position.copy()
            ret[2] = 0
            return [True, ret]
        else:
            # uses a approximiation of the inverse function in the curved region for performance reasons.
            estimated_radius = Geometry.estimator.predict(Geometry.tranformer.fit_transform(pos_rz[None, :]))[0]
            ret = Geometry.hole_origin+P_rel_O*estimated_radius + np.array([0, 0, Geometry.h(estimated_radius)])
            if np.linalg.norm(position-ret) <= radius:
                return [True, ret]
            else:
                return [False, None]


class Condition:
    # assume perfect sphere with
    def __init__(self, position, velocity, angular_velocity, time_stamp=0):
        # velocity is half a time step before position as outlined by leap frog integration
        # TODO: optimization
        self.position = position
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.time = time_stamp

        self.is_in_contact, self.contact_position = Geometry.getContact(self.position)

        self.contact_distance=-1
        if (self.is_in_contact):
            self.contact_distance=np.linalg.norm(self.position-self.contact_position)
            
            self.unit_normal = normalize(self.position-self.contact_position)
            self.unit_binormal = np.cross(self.unit_normal, normalize(velocity))
            self.unit_velocity = np.cross(self.unit_binormal, self.unit_normal)
            
            self.velocity=np.linalg.norm(self.velocity)*self.unit_velocity

    # this is a function that for some reason works okay
    def get_curvature(self):
        P_rel_O = (self.position-Geometry.hole_origin)
        P_rel_O[2] = 0
        
        r=np.linalg.norm(P_rel_O)
        P_rel_O_normalized = normalize(P_rel_O)
        
        dh1dr=Geometry.h(r,1)
        dh2dr=Geometry.h(r,2)
        
        planer_velocity = self.unit_velocity.copy()
        planer_velocity[2]=0
        planer_velocity_normalized = normalize(planer_velocity)
        
        cosa=np.dot(P_rel_O_normalized,planer_velocity_normalized)
        
        h1=dh1dr*cosa
        h2=(dh1dr/r)*(1-(cosa**2))+dh2dr*(cosa**2)
        
        curvature_contact = (h2)/((1+(h1**2))**(3/2))
        return (curvature_contact*radius)/(curvature_contact+radius)        
        
    def next(self):
        # TODO: conservation of energy

        # forces
        torque = np.array([0, 0, 0], dtype=np.float64)
        force = np.array([0, 0, 0], dtype=np.float64)

        # weight
        force += inertia*np.array([0, 0, -9.81])

        # drag
        # force+= 0.5*0.001293*(-self.velocity*abs(np.linalg.norm(self.velocity)))*0.47*pi*(radius**2)

        # normal
        if self.is_in_contact:
            
            curvature = self.get_curvature()
            
            desired_up=(np.dot(self.velocity,self.velocity)*curvature)
            normal_up=desired_up-force[2]
            normal_force=self.unit_normal*(normal_up/self.unit_normal[2])
            
            force+=normal_force

        # apply forces
        acc = force/inertia
        a_acc = torque/angular_inertia

        n_v = self.velocity+acc*time_step
        n_w = self.angular_velocity+a_acc*time_step
        n_p = self.position+self.velocity*time_step+0.5*acc*(time_step**2)

        n_condition = Condition(n_p, n_v, n_w, self.time+time_step)
        if n_condition.contact_distance==-1:
            n_condition.contact_distance=self.contact_distance
        return n_condition

    def isvalid(self):
        return (self.position[0] >= 0 and self.position[0] <= 7 and 
                self.position[1] >= 0 and self.position[1] <= 2.5 and 
                self.position[2] >= -0.2+0.443660788092*radius and self.position[2] <= 0.25)


save_file_name = os.path.join(save_folder, "save.npy")
with open(save_file_name, 'wb') as f:
    np.save(f, N)

phase_data = []

with alive_bar(N*N) as bar:
    for p_x_dex in range(N):
        # generate initial location
        initial_p_x = np.interp(p_x_dex, [0, N-1], x_intial_range)
        initial_position = np.array([initial_p_x, 1.4, radius], dtype=np.float64)

        for v_y_dex in range(N):
            # generate initial velocity
            initial_v_y = np.interp(v_y_dex, [0, N-1], y_intial_range)
            initial_velocity = np.array([0, initial_v_y, 0], dtype=np.float64)

            initial_angular_velocity = np.array([0, 0, 0], dtype=np.float64)

            condition = Condition(initial_position, initial_velocity, initial_angular_velocity)
            data = []
            while True:
                n_condition = condition.next()
                if (not n_condition.isvalid()):
                    break

                condition = n_condition

                speed = np.linalg.norm(condition.velocity)

                data.append([condition.time, *condition.position, speed, condition.contact_distance])

            data = np.array(data).T

            with open(save_file_name, 'ab') as f:
                np.save(f, data)

            phase_data.append(np.array([initial_p_x, initial_v_y, data[0, -1]]))

            bar()

phase_data = np.array(phase_data).T

with open(save_file_name, 'ab') as f:
    np.save(f, phase_data)

# import subprocess
# subprocess.call(f"python -m reader.py", shell=True)
