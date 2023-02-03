import numpy as np

def acceleration(t):
    return 0.15*(t**4)-(t**3)+(t**2)+2*t+2

t=0
dt=0.001

# first dimension left right
# second dimension up
p_hole = np.array([3.5,1.9],dtype=np.float64)

v_shoot = np.array([0,10],dtype=np.float64)
p=np.array([0,0],dtype=np.float64)
v=np.array([0,0],dtype=np.float64)
unit_a=np.array([1,0],dtype=np.float64)

while True:
    t+=dt
    v+=acceleration(t)*unit_a*dt
    p+=v*dt
    
    v_unit = (v+v_shoot) /np.linalg.norm((v+v_shoot))
    
    projected_hole = -v_unit[1]*p_hole[0]+v_unit[0]*p_hole[1]
    projected_pos = -v_unit[1]*p[0]+v_unit[1]*p[1]
    
    if abs(projected_hole-projected_pos)<0.45:
        print(p)
        print(v+v_shoot)
        break
        
    
