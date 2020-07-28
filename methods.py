import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.rcdefaults()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", figsize=(11.2, 6.3))


def LorenzSystem(sigma, rho, beta):
    def ODE(t, state):
        x, y, z = state
        return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    return ODE


def ParametricPlot(*args, labels=None, title=""):
        
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.w_xaxis.pane.set_color("w")
    ax.w_yaxis.pane.set_color("w")
    ax.w_zaxis.pane.set_color("w")
    ax.set_xlabel("x Axis")
    ax.set_ylabel("y Axis")
    ax.set_zlabel("z Axis")
    ax.set_title(f"{title}", pad=16)

    for i, states in enumerate(args):
        if labels is None:
            ax.plot(*states.T, linewidth=0.5, alpha=0.95)
        else:
            ax.plot(*states.T, linewidth=0.5, alpha=0.95, label=f"{labels[i]}")

    if labels is not None:
        plt.legend()
    plt.show()

    
def Plot(t, *args, labels=None, title="", sameAxis=True):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1, xticklabels=[])
    ax2 = fig.add_subplot(3, 1, 2, xticklabels=[])
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.set_title(f"{title}", pad=24)
    ax1.set_ylabel("x Axis")
    ax2.set_ylabel("y Axis")
    ax3.set_ylabel("z Axis")
    ax3.set_xlabel("time")
    
    
    if sameAxis:
        for i, states in enumerate(args):
            if labels is None:
                ax1.plot(t, states.T[0], color=f"C{i}", linewidth=1, alpha=0.95)
            else:
                ax1.plot(t, states.T[0], color=f"C{i}", linewidth=1, alpha=0.95, label=f"{labels[i]}")
            ax2.plot(t, states.T[1], color=f"C{i}", linewidth=1, alpha=0.95)
            ax3.plot(t, states.T[2], color=f"C{i}", linewidth=1, alpha=0.95)
    
    else:
        for i, states in enumerate(args):
            if labels is None:
                ax1.plot(t[i], states.T[0], color=f"C{i}", linewidth=1, alpha=0.95)
            else:
                ax1.plot(t[i], states.T[0], color=f"C{i}", linewidth=1, alpha=0.95, label=f"{labels[i]}")
            ax2.plot(t[i], states.T[1], color=f"C{i}", linewidth=1, alpha=0.95)
            ax3.plot(t[i], states.T[2], color=f"C{i}", linewidth=1, alpha=0.95)
    
    if labels is not None:
        ax1.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=len(args), mode="expand", borderaxespad=0)
    plt.show()


def Animate(*args, title="", xlim=[-20, 20], ylim=[-25, 25], zlim=[0, 50]):
    
    fig = plt.figure(figsize=(12.8, 7.2), tight_layout=True)
    ax = fig.gca(projection="3d")

    ax.w_xaxis.pane.set_color("w")
    ax.w_yaxis.pane.set_color("w")
    ax.w_zaxis.pane.set_color("w")
    ax.set_xlabel("x Axis")
    ax.set_ylabel("y Axis")
    ax.set_zlabel("z Axis")
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    ax.set_title(f"{title}", pad=16)

    lines = ()
    points = ()
    for i, states in enumerate(args):
        lines += ax.plot(states[0:1, 0], states[0:1, 1], states[0:1, 2], color=f"C{i}", linewidth=0.5, alpha=0.95)[0],
        points += ax.plot(states[0:1, 0], states[0:1, 1], states[0:1, 2], ".", color=f"C{i}")[0],
    plt.close()

    def init():
        for _, line, point in zip(args, lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return (*lines, *points)

    def update(i):
        for states, line, point in zip(args, lines, points):
            line.set_data(states.T[:2, :i])
            line.set_3d_properties(states.T[2, :i])
            point.set_data(states[i, :2])
            point.set_3d_properties(states[i, 2])
        return (*lines, *points)

    FuncAnimation(fig, update, frames=len(args[0]), init_func=init, interval=4, blit=True).save("Lorenz-System Animation.mp4", bitrate=5000)


def ABM(f, state0, t, steps=5, iters=1, tol=1e-8):
    '''
    returns array of states solved from the initial state using the predictor-corrector method Adams-Bashforth-Moulton

    f ........ function of the ODE: y' = f(t, y)
    state0 ... initial state vector [state0_1, state0_2, ...]
    t ........ discretized time interval [t0, t1, ...]
    steps .... number of interpolation steps per time step
    iters .... number of iterations for each correction-evaluation cycle
    tol ...... if this tolerance is met the correction-evaluation cycle breaks
    '''
    
    # input processing
    state0 = np.array(state0)
    s = steps if 0 < steps < 5 else 5
    iters = int(iters) if 0 < iters else 1
    N = len(t)
    h = (t[-1] - t[0]) / N

    # initialize state array with initial state
    states = np.zeros((N, 3))
    states[0] = state0

    # initialize ODE function values
    fvals = np.zeros((N, 3))
    fvals[0] = f(t[0], state0)

    # coefficients for Adams-Bashforth method
    coeffsAB = (1,
                [-1/2, 3/2],
                [5/12, -16/12, 23/12],
                [-9/24, 37/24, -59/24, 55/24],
                [251/720, -1274/720, 2616/720, -2774/720, 1901/720])

    # coefficients for Adams-Moulton method
    coeffsAM = (1,
                [1/2, 1/2],
                [-1/12, 2/3, 5/12],
                [1/24, -5/24, 19/24, 9/24],
                [-19/720, 106/720, -264/720, 646/720, 251/720])

    # increasing steps until desired order is reached
    for n in range(s):
        # predictor: Adams-Bashforth method
        states[n+1] = states[n] + h * np.dot(coeffsAB[n], fvals[:n+1])

        # evaluation
        fvals[n+1] = f(t[n+1], states[n+1])

        # correction-evaluation cycle with Adams-Moulton
        for _ in range(iters):
            new = states[n] + h * np.dot(coeffsAM[n], fvals[:n+1])
            cond = np.allclose(states[n+1], new, atol=tol)
            states[n+1] = new
            fvals[n+1] = f(t[n+1], states[n+1])
            if cond:
                break

    # main loop
    for n in range(s, N-1):
        # predictor: Adams-Bashforth method
        states[n+1] = states[n] + h * np.dot(coeffsAB[s-1], fvals[n-s+1:n+1])

        # evaluation
        fvals[n+1] = f(t[n+1], states[n+1])

        # correction-evaluation cycle with Adams-Moulton
        for _ in range(iters):
            new = states[n] + h * np.dot(coeffsAM[s-1], fvals[n-s+2:n+2])
            cond = np.allclose(new, states[n+1], atol=tol)
            states[n+1] = new
            fvals[n+1] = f(t[n+1], states[n+1])
            if cond:
                break

    return states