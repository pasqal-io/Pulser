import numpy as np
from qutip import *
import types

import matplotlib.pyplot as plt



sx = sigmax()
nz = (qeye(2) + sigmaz()) / 2

def Kron(V):
    """ Kroenecker product of a list of operators

    Parameters
    ----------
    V : list
        list of operators

    Returns
    -------
    numpy.ndarray
    """
    P=1
    for i in range(len(V)):
        P=np.kron(P,V[i])
    return P

def sum_oper(N,oper):
    """Compute the collective operator applying oper to each atom.

    Parameters
    ----------
    N : int
        Number of atoms
    oper : Qobj
        Operator to apply to each atom

    Returns
    -------
    Qobj:
        2**N-dim collective operator
    """
    return Qobj(sum([Kron([qeye(2**j),oper,qeye(2**(N-j-1))]) for j in range(N)]))

def H_AF(Omega,delta,U,R):
    """Ising Hamiltonian with given control functions and interaction parameters.
    If controls are scalars, compute the Qobj operator.
    If controls are functions, compute the Hamiltonian in the form needed to be fed to Qutip solver.

    Parameters
    ----------
    Omega : scalar or function
        Control Rabi frequency
    delta : scalar or function
        Control detuning
    U : float
        Interaction energy of nearest neighbor atoms
    R : numpy.ndarray
        Distance matrix of the atoms configuration
    Returns
    -------
    Qobj or list:
        Ising Hamiltonian
    """
    N=len(R)
    if isinstance(Omega,float)&isinstance(delta,float):
        H=1/2*Omega*sum_oper(N,sx)-delta*sum_oper(N,nz)
        for i in range(N):
            for j in range(N):
                Q_list=[qeye(2)]*N
                if j!=i:
                    Q_list[i]=nz
                    Q_list[j]=nz
                    H+=1/2*(U/R[i,j]**6)*Qobj(Kron(Q_list))

    elif isinstance(Omega,types.FunctionType)&isinstance(delta,types.FunctionType):
        H0=qeye(1)
        for i in range(N):
            for j in range(N):
                Q_list=[qeye(2)]*N
                if j!=i:
                    Q_list[i]=nz
                    Q_list[j]=nz
                    H0+=1/2*(U/R[i,j]**6)*Qobj(Kron(Q_list))
        H1=1/2*sum_oper(N,sx)
        H2=-1.0*sum_oper(N,nz)
        H=[H0,[H1,Omega],[H2,delta]]
    return H

def expect_val(state,oper):
    """Compute the list of expectation value of oper for a given state of N atoms

    Parameters
    ----------
    state : Qobj ket or density matrix
        State of the N atoms
    oper : Qobj operator
        One qubit operator
    Returns
    -------
    list of floats:
        list of <oper_i>
    """
    N=int(np.log2(state.shape[0]))
    expect=np.zeros((N))
    if state.isket:
        for i in range(N):
            expect[i]=(state.dag()*Qobj(Kron([qeye(2**i),oper,qeye(2**(N-i-1))]))*state).tr()
    elif state.isoper:
        for i in range(N):
            expect[i]=np.trace(state*Qobj(Kron([qeye(2**i),oper,qeye(2**(N-i-1))])))
    return expect


def var_val(state,oper):
    """Compute the covariance matrix of oper for a given state of N atoms
    Parameters
    ----------
    state : Qobj ket or density matrix
        State of the N atoms
    oper : Qobj operator
        One qubit operator
    Returns
    -------
    numpy.ndarray:
        matrix of <oper_i oper_j>
    """
    N=int(np.log2(state.shape[0]))
    var=np.zeros((N,N))
    if state.isket:
        for i in range(N):
            for j in range(N):
                Q_list=[qeye(2)]*N
                Q_list[i]=oper
                Q_list[j]=oper
                var[i,j]=(state.dag()*Qobj(Kron(Q_list))*state).tr()
    elif state.isoper:
        for i in range(N):
            for j in range(N):
                Q_list=[qeye(2)]*N
                Q_list[i]=oper
                Q_list[j]=oper
                var[i,j]=np.trace(state*Qobj(Kron(Q_list)))
    return var

def Fidelity(state,target_state):
    return (target_state.dag() * state).norm()

def stg_mag(state):
    """Compute the staggered magnetization of a given state.

    Parameters
    ----------
    state : Qobj ket or density matrix
        State of the N atoms

    Returns
    -------
    float:
        Staggred magnetization
    """
    N=int(np.log2(state.shape[0]))
    expect=expect_val(state,nz)
    return sum([(-1)**j*ex/((N+1)//2) for j,ex in enumerate(expect)])

def g2(expect,var,K):
    """Compute the correlation function for a given separation vector K

    Parameters
    ----------
    expect : list of floats:
        list of <oper_i>
    var : matrix:
        matrix of <oper_i oper_j>
    K : tuple
        Separation vector

    Returns
    -------
    float:
        g2(K)
    """
    d=len(K)
    N=len(expect)
    if K==[0]*len(K):
        return 0
    else:
        Pairs=atoms_config.nb_pairs(atoms_config.atom_list(N,d),K)
        Nkl=len(Pairs)
        G=0
        for pair in Pairs:
            i,j=pair
            G+=var[i,j]
            # -expect[i]*expect[j]
        return G/Nkl

def g2_list(state,oper,d):
    """Compute the connected spin-spin correlation for a state of N atoms on a grid of dim d
    Parameters
    ----------
    state : Qobj ket or density matrix
        State of the N atoms
    oper : Qobj operator
        One qubit operator
    d : integer
        Dimension of the grid
    Returns
    -------
    list:
        list of g2(k) if d=1 or matrix of g2(k,l) if d=2
    """
    N=int(np.log2(state.shape[0]))
    n=int(N**(1/d))
    expect=expect_val(state,oper)
    var=var_val(state,oper)
    if d==1:
        p=[[i] for i in range(-n+1,n)]
        g=np.array([g2(expect,var,e) for e in p])
    elif d==2:
        p=[[[a,b] for a in range(-n+1,n)] for b in range(-n+1,n)]
        g=np.zeros((len(p),len(p)))
        for i in range(len(p)):
            for j in range(len(p)):
                g[i,j]=g2(expect,var,p[i][j])
    return g

def S_Neel(state,d=1):
    g2=g2_list(state,sigmaz(),d)
    n=int((len(g2)+1)/2)
    S=0
    if d==1:
        for i,g in enumerate(g2):
            S+=(-1)**(abs(i-(n-1)))*g
    elif d==2:
        for i in range(len(g2)):
            for j in range(len(g2)):
                S+=(-1)**(abs(i-(n-1))+abs(j-(n-1)))*g2[i,j]
    return S/((2*n-1)**d-1)

def proba_from_state(state,rho_target=None,p=0.1,):
    """Get the states with their probability if the latter is greater than p if the state is a ket.
    Get the probability associated with rho_target if the state is a density matrix

    Parameters
    ----------
    state : Qobj ket or density matrix
        Qobj describing the state of the system
    rho_target : Qobj density matrix
        Target state
    p : float
        Minimum probability to classify a state significant

    """
    N=int(np.log2(state.shape[0]))
    if state.isket:
        proba=abs(np.array(state))**2
        indic=np.where(proba>=p)[0]
        Proba=[]
        for indi in indic:
            Proba+=['AF '+format(indi,'0{}b'.format(N))+' with P={:.4f}%'.format(proba[indi][0]*100)]
        return Proba
    elif state.isoper:
        print('AF with P={:.4f}%'.format(np.real(np.trace(rho_target*state)*100)))


def atoms_from_state(state,disp='BIN'):
    """Translate a state of the form basis(2^N,i) to a string of 0 and 1 where a 0 states for a spin up and a 1 for a spin down.
    Parameters
    ----------
    state : Qobj ket or density matrix
        State of the N atoms

    Returns
    -------
    string:
        binary string describing the state of the atom configuration
    """
    N=int(np.log2(state.shape[0]))
    indi=np.where(np.array(state)==1)[0][0]
    str=format(indi,'0{}b'.format(N))
    if disp=='BIN':
        return str
    elif disp=='SPIN':
        return str.replace('0',u'\u2191').replace('1',u'\u2193')

def measure_state(state):
    """ Simulate the measurement of a quantum state by giving a possible measured state according to the probability density of the quantum state
    Parameters
    ----------
    state : Qobj ket
        State of the N atoms

    Returns
    -------
    Qobj ket:
        Normalised state with only 0 and 1
    """
    proba=abs(np.array(state))**2
    if sum(proba)[0]!=1:
        proba=proba/sum(proba)
    bin=np.random.rand()
    proba_cumul=[0]+[sum(proba[:i+1]).tolist()[0] for i in range(len(proba))]
    indi=np.where([(bin>=proba_cumul[i]) and (bin<proba_cumul[i+1]) for  i in range(len(proba_cumul))])[0][0]

    return basis(int(len(proba)),indi)

def binom_noise_state(state,n_shots=100):
    """ Simulate the binomial noise of an experiment repeated n_shots times.
    Parameters
    ----------
    state : Qobj ket
        State of the N atoms

    Returns
    -------
    Qobj ket:
        Noisy state
    """
    return sum([measure_state(state) for i in range(n_shots)]).unit()


def binom_score(state,score,n_shots=100):
    STATE=[measure_state(state) for i in range(n_shots)]
    return sum([score(state) for state in STATE])/n_shots

def g_min(Omega,delta,R):
    H=H_AF(Omega,delta,settings_AF.U,R)
    def Hamil(t,*args):
        return H[0]+H[1][0]*H[1][1](t)+H[2][0]*H[2][1](t)
    gap=[Hamil(t).eigenstates()[0][1]-Hamil(t).eigenstates()[0][0] for t in settings_AF.time_domain]
    return min(gap)

def big_eps(Omega,delta,R):
    H=H_AF(Omega,delta,settings_AF.U,R)
    def Hamil(t,*args):
        return H[0]+H[1][0]*H[1][1](t)+H[2][0]*H[2][1](t)
    def dHamil(t,*args):
        return H[1][0]*(Omega(t+settings_AF.eps)-Omega(t))/settings_AF.eps+H[2][0]*(delta(t+settings_AF.eps)-delta(t))/settings_AF.eps
    Eps=[(Hamil(t).eigenstates()[1][1].dag()*dHamil(t)*Hamil(t).eigenstates()[1][0]).norm() for t in settings_AF.time_domain]
    return max(Eps)

def plot_g2(g2_list):
    n=len(g2_list)//2+1
    d=len(g2_list[0])
    plt.figure()
    plt.imshow(np.atleast_2d(g2_list), cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('k')
    plt.yticks([])
    plt.xticks(range(len(g2_list)), ['{}'.format(i) for i in range(-n + 1, n)])
    if d>1:
        plt.yticks(range(len(g2_list)), ['{}'.format(i) for i in np.arange(n - 1, -n, -1)])
        plt.title(r'$g_2(k,l)$')
        plt.ylabel('l')
    plt.colorbar()
    plt.show()