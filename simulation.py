from math import sqrt, pi
import numpy as np
from alive_progress import alive_bar
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os

# material property
radius = 0.02
density = 1000
friction_coefficient = 0.3
rolling_friction_coefficient = 0.00001

restore_ratio = 0

# simulation property
time_step = 0.001
tol = 0.000_001
N = 20
x_intial_range = [3.00, 3.5]
y_intial_range = [0.5, 0.75]

# save property
save_folder = "results"


# pre computation
inertia = (4/3)*pi*(radius**3)*density
angular_inertia = (2/5)*(inertia*radius**2)

save_file_name=os.path.join(save_folder,"save.npy")

estimator = pickle.load(open("estimator.sav", 'rb'))
tranformer = PolynomialFeatures(20)


class Geometry:
    hole_origin = np.array([3.5, 1.9, 0])

    @staticmethod
    def h(r):
        if r <= 0.1272:
            return -1
        elif r <= 0.15:
            return -0.457+2.02*r
        elif r <= 0.45:
            # return -((r**2)-0.913*r+0.209)/(4.93*(r**2)+r+0.36)
            return (0.154/0.15134222365)*(-((r**2)-(0.913240567396*r)+0.208458255328)/(4.93*(r**2)+r+0.36))
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
        if (position[2] > radius+tol):
            return [False, None]
        # first switch to rz

        P_rel_O = (position-Geometry.hole_origin)
        P_rel_O[2] = 0

        pos_rz = np.array([np.linalg.norm(P_rel_O), position[2]])
        P_rel_O = P_rel_O/np.linalg.norm(P_rel_O)

        # check regions
        if pos_rz[1]<-0.2-(radius+tol):
            # definitely below the well
            return [False,None]
        elif pos_rz[1]<-0.49504950495*pos_rz[0]-0.13708570297:
            # almost below the well
            if (np.linalg.norm(pos_rz - np.array([0.1272,-0.2]))<=radius+tol):
                ret = Geometry.hole_origin+P_rel_O*0.1272+np.array([0,0,-0.2])
                return [True,ret]
            else:
                return [False, None]
        elif pos_rz[1]<-0.49504950495*pos_rz[0]-0.0797425742574:
            # in the straight edge
            if pos_rz[1]<2.02*pos_rz[0]-0.457+(radius+tol)*2.25397426782:
                # is close enough
                contact_radius= (0.49504950495*pos_rz[0]+0.457+pos_rz[1])/(2.51504950495)
                ret = Geometry.hole_origin+P_rel_O*contact_radius+np.array([0,0,Geometry.h(contact_radius)])
                return [True,ret]
            else:
                # is not close enough
                return [False, None]
        elif pos_rz[1]<-0.61735918881*pos_rz[0]-0.0613961216765:
            # between two wells
            if (np.linalg.norm(pos_rz - np.array([0.15,-0.154]))<=radius+tol):
                ret = Geometry.hole_origin+P_rel_O*0.15+np.array([0,0,-0.154])
                return [True,ret]
            else:
                return [False, None]
        elif pos_rz[0]>0.45:
            # out of the well region
            ret=position.copy()
            ret[2]=0
            return [True,ret]
        else:
            # uses a approximiation of the inverse function in the curved region for performance reasons.
            estimated_radius = estimator.predict(tranformer.fit_transform(pos_rz[None,:]))[0]
            ret = Geometry.hole_origin+P_rel_O*estimated_radius+np.array([0,0,Geometry.h(estimated_radius)])
            if np.linalg.norm(position-ret)<=radius+tol:
                return [True,ret]
            else:
                return [False,None]
            
            # # the following approximiates the contact point everytime it is called.
            # def error_func(r):
            #     return ((r-pos_rz[0])**2)+((Geometry.h(r)-pos_rz[1])**2)
            # minimum = fmin(error_func, pos_rz[0] +
            #                radius, disp=False, xtol=roc_step)[0]
            # if (error_func(minimum) > radius*radius):
            #     return [False, None]
            # min_pos_xyz = Geometry.hole_origin+P_rel_O*minimum
            # return [True, Geometry.GetPosition(min_pos_xyz)]
            pass


class Condition:
    # assume perfect sphere with
    def __init__(self, position, velocity, angular_velocity, time_stamp=0):
        # velocity is half a time step before position as outlined by leap frog integration
        # TODO: optimization
        self.position = position
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.time = time_stamp

        self.is_contact, self.contact_position = Geometry.getContact(
            self.position)
        if (self.is_contact):
            normal = self.position-self.contact_position
            self.unit_normal = normal / np.linalg.norm(normal)
            unit_velocity = self.velocity/np.linalg.norm(self.velocity)
            self.unit_binormal = np.cross(self.unit_normal, unit_velocity)
            self.unit_velocity = np.cross(self.unit_binormal, self.unit_normal)

            self.vbn_to_xyz = np.array([self.unit_velocity, self.unit_binormal, self.unit_normal]).T
            self.xyz_to_vbn = np.linalg.inv(self.vbn_to_xyz)

            self.vbn = np.matmul(self.xyz_to_vbn, self.contact_position)

    def next(self):
        # TODO: conservation of energy
        
        # gravity
        torque = np.array([0, 0, 0], dtype=np.float64)
        force = np.array([0, 0, 0], dtype=np.float64)

        # weight
        force += inertia*np.array([0, 0, -9.81])

        # drag
        # force+= 0.5*0.001293*(-self.velocity*abs(np.linalg.norm(self.velocity)))*0.47*pi*(radius**2)

        if self.is_contact:
            # TODO: curvature

            # the following is the amount of force that should be in the normal direction of the surface.
            # for positive curvature, this is a positive numer
            force_out_plane_theoretical = 0

            force_out_plane_current = np.dot(force, self.unit_normal)

            # if the required force in normal direction is larger than current,
            # this assume 0 curvature. additional curvature is controlled in the collision section.
            if (force_out_plane_theoretical > force_out_plane_current):
                normal_force_magnitude = (
                    force_out_plane_theoretical-force_out_plane_current)
                normal_force = self.unit_normal * normal_force_magnitude
                force += normal_force

        # force and acceleration first
        acc = force/inertia
        a_acc = torque/angular_inertia

        n_v = self.velocity+acc*time_step
        n_w = self.angular_velocity+a_acc*time_step
        n_p = self.position+self.velocity*time_step

        # deal with collision, create intermediate state
        n_condition_1 = Condition(n_p, n_v, n_w, self.time+time_step)

        # if no collision, no worries
        if not n_condition_1.is_contact:
            return n_condition_1

        # collision, move ball away from collision
        n_p = n_condition_1.unit_normal*radius+n_condition_1.contact_position

        # collision, turn the ball's velocity to be normal to surface normal
        n_v = n_condition_1.unit_velocity * \
            np.linalg.norm(n_condition_1.velocity)
        n_condition_2 = Condition(n_p, n_v, n_w, self.time+time_step)
        if(not n_condition_2.is_contact):
            Geometry.getContact(n_p)
        return n_condition_2

    def isvalid(self):
        return (self.position[0] >= 0 and self.position[0] <= 7 and self.position[1] >= 0 and self.position[1] <= 2.5 and self.position[2] >= -0.25 and self.position[2] <= 0.25)


with open(save_file_name, 'wb') as f:
    np.save(f, N)

phase_data = []

with alive_bar(N*N) as bar:
    for p_x_dex in range(N):
        initial_p_x = np.interp(p_x_dex, [0, N-1], x_intial_range)
        initial_position = np.array(
            [initial_p_x, 1.4, radius], dtype=np.float64)
        for v_y_dex in range(N):
            initial_v_y = np.interp(v_y_dex, [0, N-1], y_intial_range)
            initial_velocity = np.array([0, initial_v_y, 0], dtype=np.float64)
            initial_angular_velocity = np.array([0, 0, 0], dtype=np.float64)

            condition = Condition(initial_position, initial_velocity, initial_angular_velocity)
            st = -1
            data = []
            while True:
                n_condition = condition.next()
                if (not n_condition.isvalid()):
                    break

                condition = n_condition
                if (st == -1 and np.linalg.norm(condition.position[0:2]-np.array([3.5, 1.9])) < 0.45):
                    st = condition.time
                speed = np.linalg.norm(condition.velocity)
                is_in_contact = 1 if condition.is_contact else 0
                data.append(
                    [condition.time, *condition.position, speed, is_in_contact])

            data = np.array(data).T

            with open(save_file_name, 'ab') as f:
                np.save(f, data)

            phase_data.append(np.array([initial_p_x, initial_v_y, data[0, -1]-st]))

            bar()

phase_data = np.array(phase_data).T

with open(save_file_name, 'ab') as f:
    np.save(f, phase_data)

# import subprocess
# subprocess.call(f"python -m reader.py", shell=True)
