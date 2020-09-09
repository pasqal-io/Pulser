from pasqalobj import *
########################################

def build_init_state(N):
    """
    Builds the initial state psi_0. In this case as 'all down' except for some
    """
    state = [basis(2,0) for _ in range(N)]
    state[0] = basis(2,0) + 0.3 * basis(2,1)
    return tensor(state)

# Define some functions for the time-dependent parameters
def f_delta(t,args=2*np.pi/7):
    return np.sin(t)

def f_omega(t,args=0):
    return np.cos(t)

############################################

# Set up the graph:
N = 8
coords = MAXCOORD * np.random.rand(N,2)
# Set an observable:
site = 6 # choose some site to place the observable
op_list = [qeye(2) for _ in range(N)]
op_list[site] = sigmaz()
observable = tensor(op_list)

# Initial state and time array
psi_0 = build_init_state(N)
t = np.linspace(0.0, 10.0, 200)


# Now we define our time-dependent problem:

XY = XYPasqalObj(coords, f_delta, f_omega)
ising = IsingPasqalObj(coords, f_delta, f_omega)


# First for the XY model:
H0, HX, HZ = XY.build_hamiltonian(coords, C3 = 30)
H = [H0, [HX, XY.omega], [HZ, XY.delta]]
# Solve time evolution:
output_XY = sesolve(H, psi_0, t, [observable])

# Now for the Ising Model
iH0, iHX, iHZ = ising.build_hamiltonian(coords, C6 = 30)
iH = [iH0, [iHX, ising.omega], [iHZ, ising.delta]]
# Solve time evolution:
output_ising = sesolve(iH, psi_0, t, [observable])


plot_evol = plt.figure(1)
plt.plot(t,output_XY.expect[0])
#plot_evol2 = plt.figure(2)
plt.plot(t,output_ising.expect[0])


plot_graph = plt.figure(0)
plt.plot(coords[:,0], coords[:,1], "o")


plt.show()
