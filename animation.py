import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
plt.style.use("seaborn-whitegrid")
plt.rcParams["figure.figsize"] = [12.8, 7.2]

class LorenzSystem:
    
    def __init__(self, rho, r, b):
        '''initialize the object with the parameters rho, r and b of the system'''
        
        self.rho = rho
        self.r = r
        self.b = b
                
    def ODE(self, state):
        '''differential equation of the object: x'=rho*(x-y), y'=x*(r-z)-y, z'=x*y-b*z'''
        
        x, y, z = state
        return np.array([self.rho*(x-y), x*(self.r-z)-y, x*y-self.b*z])
    
    def solve(self, state0, t, steps=4):
        '''solves the ODE with given initial state and discretized time interval using ABE'''
        
        state0 = np.array(state0)
        self.t = np.array(t)
        N = len(t)-1
        h = (t[-1] - t[0]) / N
        
        # initialize state array with initial state
        states = np.zeros((N+1, 3))
        states[0] = state0
        
        # initialize ODE function values
        fvals = np.zeros((N+1, 3))
        fvals[0] = self.ODE(state0)
        
        # coefficients for Adams-Bashforth method
        coeffsAB = (1,
                    [-1/2, 3/2],
                    [5/12, -16/12, 23/12],
                    [-9/24, 37/24, -59/24, 55/24])
        
        # coefficients for Adams-Moulton method
        coeffsAM = ([1/2, 1/2],
                    [-1/12, 2/3, 5/12],
                    [1/24, -5/24, 19/24, 9/24],
                    [-19/720, 106/720, -264/720, 646/720, 251/720])
        
        # increasing steps until desired order is reached
        for s in range(steps):
            # predictor: Adams-Bashforth method
            states[s+1] = states[s] + h * np.dot(coeffsAB[s], fvals[:s+1])
            
            # corrector: Adams-Moulton method
            states[s+1] = states[s] + h * (np.dot(coeffsAM[s][:-1], fvals[:s+1]) + coeffsAM[s][-1] * self.ODE(states[s+1]))
            
            # evaluation
            fvals[s+1] = self.ODE(states[s+1])
        
        # main loop
        for n in range(N-steps+1):
            # predictor: Adams-Bashforth method
            states[n+steps] = states[n+steps-1] + h * np.dot(coeffsAB[steps-1], fvals[n:n+steps])

            # corrector: Adams-Moulton method
            states[n+steps] = states[n+steps-1] + h * (np.dot(coeffsAM[steps-1][:-1], fvals[n:n+steps]) + coeffsAM[steps-1][-1] * self.ODE(states[n+steps]))
            
            # evaluation
            fvals[n+steps] = self.ODE(states[n+steps])
        
        self.states = states
        return states
    
    def plot(self):
        '''plots the solved system on a 3D axis'''
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot(self.states[:, 0], self.states[:, 1], self.states[:, 2])
        plt.show()

    def saveplot(self, str):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_xlim(-20,20)
        ax.set_ylim(-25,25)
        ax.set_zlim(0,50)
        ax.plot(self.states[:, 0], self.states[:, 1], self.states[:, 2])
        plt.title(f"rho = {self.rho:5.1f}, r = {self.r:5.1f}, b = {self.b:5.1f}")
        plt.savefig(str, dpi=72, bbox_inches="tight")
        plt.clf()
        plt.close()

            
rho = -10
r = 28
b = 8 / 3

state0 = [1, 1, 1]
t = np.linspace(0, 100, 10_000)

def animateRho(frames):
    for i, Rho in enumerate(frames):
        LS = LorenzSystem(Rho, r, b)
        LS.solve(state0, t)
        LS.saveplot(f"frames/frame{i}.png")

def animateR(frames):
    for i, R in enumerate(frames):
        LS = LorenzSystem(rho, R, b)
        LS.solve(state0, t)
        LS.saveplot(f"frames/frame{i}.png")

def animateB(frames):
    for i, B in enumerate(frames):
        LS = LorenzSystem(rho, r, B)
        LS.solve(state0, t)
        LS.saveplot(f"frames/frame{i}.png")

#animateRho(np.linspace(rho+5, rho-5, 240))
#animateR(np.linspace(r+5, r-5, 240))
#animateB(np.linspace(b+5, b-5, 240))