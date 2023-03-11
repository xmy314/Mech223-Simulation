import numpy as np
import sympy as sp
import pickle
from sklearn.preprocessing import PolynomialFeatures

# this file is created entirely to hide the amount of magic numbers in this piece of code


class Well:
    hole_origin = np.array([0, 0, 0])
    estimator = pickle.load(open("estimator.sav", 'rb'))
    tranformer = PolynomialFeatures(20)

    def __init__(self, ball_radius) -> None:
        self.ball_radius = ball_radius

        r = sp.symbols("r")
        self.symbolic_r = r
        # self.h0 =  -((r**2)-0.913*r+0.209)/(4.93*(r**2)+r+0.36)
        # this is a slightly modified profile to fix the discontinuity of the curve.
        self.h0 = (0.154/0.15134222365)*(-((r**2)-(0.913240567396*r)+0.208458255328)/(4.93*(r**2)+r+0.36))
        self.h1 = sp.diff(self.h0, r)
        self.h2 = sp.diff(self.h0, r)

        self.H = sp.lambdify((r), self.h0, "numpy")
        self.H1 = sp.lambdify((r), self.h1, "numpy")
        self.H2 = sp.lambdify((r), self.h2, "numpy")

    # r is radius from the center of the well
    # order is the nth order derivative.
    def h(self, r, order=0):
        if isinstance(r, np.ndarray):
            h = np.zeros_like(r)
            range_0 = np.all([r <= 0.15], axis=0)
            range_1 = np.all([0.15 < r, r <= 0.45], axis=0)
            range_2 = np.all([0.45 < r], axis=0)
            if order == 0:
                h[range_0] = -0.457+2.02*r[range_0]
                h[range_1] = self.H(r[range_1])
                h[range_2] = 0
            elif order == 1:
                h[range_0] = 2.02
                h[range_1] = self.H1(r[range_1])
                h[range_2] = 0
            elif order == 2:
                h[range_0] = 0
                h[range_1] = self.H1(r[range_1])
                h[range_2] = 0
            return h
        else:
            if order == 0:
                if r <= 0.15:
                    return -0.457+2.02*r
                elif r <= 0.45:
                    return float(self.h0.subs(self.symbolic_r, r))
                else:
                    return 0
            elif order == 1:
                if r <= 0.15:
                    return 2.02
                elif r <= 0.45:
                    return float(self.h1.subs(self.symbolic_r, r))
                else:
                    return 0
            elif order == 2:
                if r <= 0.15:
                    return 0
                elif r <= 0.45:
                    return float(self.h2.subs(self.symbolic_r, r))
                else:
                    return 0

    def GetPosition(self, position_xy0):
        P_rel_O = (position_xy0-Well.hole_origin)
        P_rel_O[2] = 0
        height = self.h(np.linalg.norm(P_rel_O))
        position_xyz = np.array([position_xy0[0], position_xy0[1], height])
        return position_xyz

    # numerically approximiate the point on surface closed to the ball
    # this function is an abomination
    # each of the if statement refers to a single section of the curve or a junction of two sections of the curve.
    def getContact(self, position):
        # first switch to rz

        P_rel_O = (position-Well.hole_origin)
        P_rel_O[2] = 0

        pos_rz = np.array([np.linalg.norm(P_rel_O), position[2]])
        P_rel_O = P_rel_O/np.linalg.norm(P_rel_O)

        # check regions
        if pos_rz[1] < -0.2-(self.ball_radius):
            # definitely below the well
            return [False, None]
        elif pos_rz[1] < -0.49504950495*pos_rz[0]-0.13708570297:
            # almost below the well
            ret = Well.hole_origin+P_rel_O * 0.1272+np.array([0, 0, -0.2])
            if (np.linalg.norm(pos_rz - np.array([0.1272, -0.2])) <= self.ball_radius):
                return [True, ret]
            else:
                return [False, ret]
        elif pos_rz[1] < -0.49504950495*pos_rz[0]-0.0797425742574:
            # in the straight edge
            contact_radius = (0.49504950495*pos_rz[0]+0.457+pos_rz[1])/(2.51504950495)
            ret = Well.hole_origin+P_rel_O*contact_radius + np.array([0, 0, self.h(contact_radius)])
            if pos_rz[1] < 2.02*pos_rz[0]-0.457+(self.ball_radius)*2.25397426782:
                # is close enough
                return [True, ret]
            else:
                # is not close enough
                return [False, ret]
        elif pos_rz[1] < -0.61735918881*pos_rz[0]-0.0613961216765:
            # between two wells
            ret = Well.hole_origin+P_rel_O * 0.15+np.array([0, 0, -0.154])
            if (np.linalg.norm(pos_rz - np.array([0.15, -0.154])) <= self.ball_radius):
                return [True, ret]
            else:
                return [False, ret]
        elif pos_rz[0] > 0.45:
            # out of the well region
            ret = position.copy()
            ret[2] = 0
            return [True, ret]
        else:
            # uses a approximiation of the inverse function in the curved region for performance reasons.
            estimated_radius = Well.estimator.predict(Well.tranformer.fit_transform(pos_rz[None, :]))[0]
            ret = Well.hole_origin+P_rel_O*estimated_radius + np.array([0, 0, self.h(estimated_radius)])
            if np.linalg.norm(position-ret) <= self.ball_radius:
                return [True, ret]
            else:
                return [False, ret]
