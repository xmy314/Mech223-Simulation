import numpy as np    
import matplotlib.pyplot as plt
from matplotlib import ticker
from math import sqrt,pi
import os

save_folder = "results"
save_file_name=os.path.join(save_folder,"save.npy")

def h(r):
    ret=np.zeros_like(r)
    ret[r<0.45]=-((r[r<0.45]**2)-0.913*r[r<0.45]+0.209)/(4.93*(r[r<0.45]**2)+r[r<0.45]+0.36)
    ret[r<0.15]=-0.457+2.02*r[r<0.15]
    ret[r<0.1272]=-1
    return ret
def plot_well(plot_handle):
    r, theta = np.meshgrid(np.linspace(0.1273,0.45,10), np.linspace(0, 2*pi, 6*4+1))
    x,y,z = r*np.cos(theta)+3.5,r*np.sin(theta)+1.9,np.reshape(h(r.flatten()),r.shape)
    plot_handle.plot_wireframe(x,y,z,zorder=1,linewidths=0.5)
    

fig=plt.figure(figsize=(12,8))
seqential_graph = fig.add_subplot(2,2,1,projection='3d')
phase_graph = fig.add_subplot(2,2,2)
energy_graph = fig.add_subplot(2,2,3)
probe_graph = fig.add_subplot(2,2,4)


plot_well(seqential_graph)


with open(save_file_name, 'rb') as f:
    N = np.load(f)
    try: # this is to read from early stopped cases
    
        for i in range(N**2):
            t,x,y,z,speed,contact = np.load(f)
            X=i//N
            Y=i%N
            if np.linalg.norm(np.array([x[-1],y[-1]])-np.array([3.5,1.9]))>0.45:
                continue
            if t[-1]<=1:
                continue
            seqential_graph.plot(x, y, z)
            g_energy=9.81*z
            k_energy=0.5*speed**2
            energy_graph.plot(t,g_energy+k_energy)
            probe_graph.plot(t[:-1],contact[:-1]/1000 )
            # probe_graph.plot(t[:-1],k_energy[:-1]-k_energy[1:]+g_energy[:-1]-g_energy[1:] )
            # probe_graph.plot(k_energy+g_energy,z)
            
        px,vy,t = np.load(f)
        px=px.reshape((N,N))
        vy=vy.reshape((N,N))
        t=t.reshape((N,N))
        cs=phase_graph.contourf(px, vy, t,
                            locator = ticker.MaxNLocator(30),
                            cmap ="cool")
        cbar = plt.colorbar(cs)
        
    except:
        pass
        

seqential_graph.set_xlim([3, 4])
seqential_graph.set_ylim([1.4, 2.4])
seqential_graph.set_zlim([-0.5, 0.1])

seqential_graph.set_xlabel("X")
seqential_graph.set_ylabel("Y")
seqential_graph.set_zlabel("Z")

phase_graph.set_xlabel("position")
phase_graph.set_ylabel("velocity")


energy_graph.set_xlabel("time")
energy_graph.set_ylabel("energy")

probe_graph.set_xlabel("energy")
probe_graph.set_ylabel("elevation")

fig.tight_layout()
plt.show()