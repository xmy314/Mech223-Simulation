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
        a = sp.symbols("a")  # this stands for angle. it is used because theta is not a single character.

        self.symbolic_r = r
        self.symbolic_a = a

        h_straight = -0.457+2.02*r
        # this is a slightly modified profile to fix the discontinuity of the curve.
        # h_curve =  -((r**2)-0.913*r+0.209)/(4.93*(r**2)+r+0.36)
        h_curve = (0.154/0.15134222365)*(-((r**2)-(0.913240567396*r)+0.208458255328)/(4.93*(r**2)+r+0.36))
        h_flat = 0
        self.h0 = sp.Piecewise(
            (h_straight, r <= 0.15),
            (h_curve, r <= 0.45),
            (h_flat, True)
        )
        self.h1 = sp.diff(self.h0, r)
        self.h2 = sp.diff(self.h1, r)

        length_inverse_scaling_factor = 1/sp.sqrt(1+(self.h1)**2)

        # parametrically define the surface that contains the ball
        # assume smooth surface which isn't true.

        # n stands for upward normal vector
        self.nx = sp.cos(a)*(-self.h1*length_inverse_scaling_factor)
        self.ny = sp.sin(a)*(-self.h1*length_inverse_scaling_factor)
        self.nz = (length_inverse_scaling_factor)

        # b stands for ball center
        self.bx = sp.cos(a)*r+self.ball_radius*self.nx
        self.by = sp.sin(a)*r+self.ball_radius*self.ny
        self.bz = self.h0+self.ball_radius*self.nz

        # first derivative with respect to unwrapped coordinates
        self.bx_r = sp.diff(self.bx, r)
        self.by_r = sp.diff(self.by, r)
        self.bz_r = sp.diff(self.bz, r)
        self.bx_a = sp.diff(self.bx, a)
        self.by_a = sp.diff(self.by, a)
        self.bz_a = sp.diff(self.bz, a)

        # second derivative with respect to unwrapped coordinates
        self.bx_rr = sp.diff(self.bx_r, r)
        self.by_rr = sp.diff(self.by_r, r)
        self.bz_rr = sp.diff(self.bz_r, r)
        self.bx_ar = sp.diff(self.bx_a, r)
        self.by_ar = sp.diff(self.by_a, r)
        self.bz_ar = sp.diff(self.bz_a, r)
        self.bx_ra = sp.diff(self.bx_r, a)
        self.by_ra = sp.diff(self.by_r, a)
        self.bz_ra = sp.diff(self.bz_r, a)
        self.bx_aa = sp.diff(self.bx_a, a)
        self.by_aa = sp.diff(self.by_a, a)
        self.bz_aa = sp.diff(self.bz_a, a)

        # lambify everything for number crunching
        self.H0 = sp.lambdify((r), self.h0, "numpy")
        self.H1 = sp.lambdify((r), self.h1, "numpy")
        self.H2 = sp.lambdify((r), self.h2, "numpy")

        self.NX = sp.lambdify((r, a), self.nx, "numpy")
        self.NY = sp.lambdify((r, a), self.ny, "numpy")
        self.NZ = sp.lambdify((r, a), self.nz, "numpy")

        self.BX = sp.lambdify((r, a), self.bx, "numpy")
        self.BY = sp.lambdify((r, a), self.by, "numpy")
        self.BZ = sp.lambdify((r, a), self.bz, "numpy")

        self.BX_R = sp.lambdify((r, a), self.bx_r, "numpy")
        self.BY_R = sp.lambdify((r, a), self.by_r, "numpy")
        self.BZ_R = sp.lambdify((r, a), self.bz_r, "numpy")
        self.BX_A = sp.lambdify((r, a), self.bx_a, "numpy")
        self.BY_A = sp.lambdify((r, a), self.by_a, "numpy")
        self.BZ_A = sp.lambdify((r, a), self.bz_a, "numpy")

        self.BX_RR = sp.lambdify((r, a), self.bx_rr, "numpy")
        self.BY_RR = sp.lambdify((r, a), self.by_rr, "numpy")
        self.BZ_RR = sp.lambdify((r, a), self.bz_rr, "numpy")
        self.BX_AR = sp.lambdify((r, a), self.bx_ar, "numpy")
        self.BY_AR = sp.lambdify((r, a), self.by_ar, "numpy")
        self.BZ_AR = sp.lambdify((r, a), self.bz_ar, "numpy")
        self.BX_RA = sp.lambdify((r, a), self.bx_ra, "numpy")
        self.BY_RA = sp.lambdify((r, a), self.by_ra, "numpy")
        self.BZ_RA = sp.lambdify((r, a), self.bz_ra, "numpy")
        self.BX_AA = sp.lambdify((r, a), self.bx_aa, "numpy")
        self.BY_AA = sp.lambdify((r, a), self.by_aa, "numpy")
        self.BZ_AA = sp.lambdify((r, a), self.bz_aa, "numpy")

    # r is radius from the center of the well
    # order is the nth order derivative with respect to r
    def h(self, r, order=0):
        if order == 0:
            return self.H0(r)
        elif order == 1:
            return self.H1(r)
        elif order == 2:
            return self.H2(r)

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

    # parametrization
    # in the following functions,
    #   ra describes the polar coordinate of the contact point.
    #       this is a parametrization of the surface.
    #   xyz describes the center of the sphere in cartesian coordinate.
    #   cxyz describes the contact point in cartesian coordinate.
    #   *dt is first order derivative respect to time
    #   *dtt is second order derivative respect to time

    def cxyz_to_ra(self, Cxyz):
        a = np.arctan2(Cxyz[1], Cxyz[0])
        r = np.linalg.norm(Cxyz[:2])
        return r, a

    def ra_to_xyz(self, contact_r, contact_a):
        # convert from parametrization space to xyz space
        return np.stack(
            [
                self.BX(contact_r, contact_a),
                self.BY(contact_r, contact_a),
                self.BZ(contact_r, contact_a)],
            axis=0)

    def xyzdt_to_radt(self, contact_r, contact_a, xyzdt):
        # convert velocity from xyz space to parametrization space
        transformation_matrix = [
            [self.BX_R(contact_r, contact_a), self.BX_A(contact_r, contact_a)],
            [self.BY_R(contact_r, contact_a), self.BY_A(contact_r, contact_a)],
            # [self.BZ_R(contact_r, contact_a), self.BZ_A(contact_r, contact_a)],
        ]
        rdt, adt = np.linalg.solve(transformation_matrix, xyzdt[:2])

        return rdt, adt

    def getNormalAcceleration(self, cxyz, xyzdt):
        # calculate acceleration normal to the surface needed to stay on the surface
        r, a = self.cxyz_to_ra(cxyz)

        rdt, adt = self.xyzdt_to_radt(r, a, xyzdt)
        goal = [
            self.BX_RR(r, a)*(rdt**2)+self.BX_RA(r, a)*(rdt*adt)+self.BX_AR(r, a)*(rdt*adt)+self.BX_AA(r, a)*(adt**2),
            self.BX_RR(r, a)*(rdt**2)+self.BX_RA(r, a)*(rdt*adt)+self.BX_AR(r, a)*(rdt*adt)+self.BX_AA(r, a)*(adt**2),
            self.BX_RR(r, a)*(rdt**2)+self.BX_RA(r, a)*(rdt*adt)+self.BX_AR(r, a)*(rdt*adt)+self.BX_AA(r, a)*(adt**2),
        ]

        transformation_matrix = [
            [self.NX(r, a), -self.BX_R(r, a), -self.BX_A(r, a)],
            [self.NY(r, a), -self.BY_R(r, a), -self.BY_A(r, a)],
            [self.NZ(r, a), -self.BZ_R(r, a), -self.BZ_A(r, a)],
        ]

        acceleration, _, _ = np.linalg.solve(transformation_matrix, goal)

        return acceleration
