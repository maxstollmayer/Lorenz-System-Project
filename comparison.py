import numpy as np
from time import process_time

def forwardEuler(f, state0, t):
    '''
    Returns array of states approximated with the forward Euler method.

    f ........ function of ODE state'(t) = f(t, state(t))
    state0 ... initial value state(t0) = state0
    t ........ discretized interval [t0, t1, ...]
    '''

    N = len(t)
    dim = (N,) + np.shape(state0)

    states = np.zeros(dim)
    states[0] = state0

    for n in range(N - 1):
        h = t[n + 1] - t[n]
        states[n + 1] = states[n] + h * f(t[n], states[n])

    return states


def fixedPointIter(f, x0, *args, tol=1e-8, iters=10000):
    '''
    Returns approximated fixed point of given function if found using a simple iteration.

    f ....... function with iterator as first positional argument
    x0 ...... initial value of iteration
    *args ... pass-through arguments of f
    tol ..... tolerance of approximation to stop iterating
    iters ... maximum number of iterations before divergence is declared
    '''

    dim = (iters,) + np.shape(x0)

    x = np.zeros(dim)
    x[0] = x0

    for i in range(iters):
        x[i + 1] = f(x[i], *args)

        if np.allclose(x[i + 1], x[i], atol=tol):
            return x[i + 1]

    else:
        print(f"Fixed-point iteration did not converge in {iters} iterations. Returned last value.")
        return x[i + 1]


def backwardEuler(f, state0, t, tol=1e-8, iters=10000):
    '''
    Returns array of states approximated with the backward Euler method using a fixed point iteration.

    f ........ function of ODE state'(t) = f(t, state(t))
    state0 ... initial value state(t0) = state0
    t ........ discretized interval [t0, t1, ...]
    tol ...... tolerance of approximation to stop iterating
    iters .... maximum number of iterations before divergence is declared
    '''

    N = len(t)
    dim = (N,) + np.shape(state0)

    states = np.zeros(dim)
    states[0] = state0

    for n in range(N - 1):
        h = t[n + 1] - t[n]

        def g(x):
            return states[n] + h * f(t[n], x)

        states[n + 1] = fixedPointIter(g, states[n], tol=tol, iters=iters)

    return states


def trapezoidalRule(f, state0, t, tol=1e-8, iters=10000):
    '''
    Returns array of states approximated with the trapezoidal rule using a fixed point iteration.

    f ........ function of ODE state'(t) = f(t, state(t))
    state0 ... initial value state(t0) = state0
    t ....... discretized interval [t0, t1, ...]
    tol ..... tolerance of approximation to stop iterating
    iters ... maximum number of iterations before divergence is declared
    '''

    N = len(t)
    d = len(state0)

    states = np.zeros((N, d))
    states[0] = state0

    for n in range(N - 1):
        h = t[n + 1] - t[n]

        def g(x):
            return states[n] + h * (f(t[n], states[n]) + f(t[n + 1], x)) / 2

        states[n + 1] = fixedPointIter(g, states[n], tol=tol, iters=iters)

    return states


def RungeKutta4(f, state0, t):
    '''
    Returns array of states approximated with the Runge-Kutta method of 4th order.

    f ........ function of ODE state'(t) = f(t, state(t))
    state0 ... initial value state(t0) = state0
    t .... discretized interval [t0, t1, ...]
    '''

    N = len(t)
    dim = (N,) + np.shape(state0)

    states = np.zeros(dim)
    states[0] = state0

    for n in range(N - 1):
        h = t[n + 1] - t[n]

        k1 = f(t[n], states[n])
        k2 = f(t[n] + h / 2, states[n] + h / 2 * k1)
        k3 = f(t[n] + h / 2, states[n] + h / 2 * k2)
        k4 = f(t[n] + h, states[n] + h * k3)

        states[n + 1] = states[n] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return states


def ABM(f, state0, t, steps=4, iters=1, tol=1e-8):
    '''
    Returns array of states solved using the predictor-corrector method: Adams-Bashforth-Moulton.

    f ........ function of the ODE: y' = f(t, y)
    state0 ... initial state vector [state0_1, state0_2, ...]
    t ........ discretized time interval [t0, t1, ...]
    steps .... number of interpolation steps per time step
    iters .... number of iterations for each correction-evaluation cycle
    tol ...... if this tolerance is met the correction-evaluation cycle stops
    '''

    # input processing
    s = int(steps) if 1 <= steps <= 4 else 4
    iters = int(iters) if 0 < iters else 1
    N = len(t)
    d = len(state0)

    # initialize state array with initial states
    states = np.zeros((N, d))
    if s == 1:
        states[0] = state0
    else:
        states[:s] = RungeKutta4(f, state0, t[:s])

    # initialize ODE function values
    fvals = np.zeros((N, d))
    for n in range(s):
        fvals[n] = f(t[n], states[n])

    # coefficients for Adams-Bashforth method
    coeffsAB = (1,
                [-1 / 2, 3 / 2],
                [5 / 12, -16 / 12, 23 / 12],
                [-9 / 24, 37 / 24, -59 / 24, 55 / 24])

    # coefficients for Adams-Moulton method
    coeffsAM = ([1 / 2, 1 / 2],
                [-1 / 12, 2 / 3, 5 / 12],
                [1 / 24, -5 / 24, 19 / 24, 9 / 24],
                [-19 / 720, 106 / 720, -264 / 720, 646 / 720, 251 / 720])

    # prediction-correction cycle
    for n in range(s - 1, N - 1):
        h = t[n + 1] - t[n]

        # predictor: Adams-Bashforth method
        states[n + 1] = states[n] + h * np.dot(coeffsAB[s - 1], fvals[n - s + 1:n + 1])

        # evaluation
        fvals[n + 1] = f(t[n + 1], states[n + 1])

        # correction-evaluation cycle with Adams-Moulton
        for _ in range(iters):
            new = states[n] + h * np.dot(coeffsAM[s - 1], fvals[n - s + 1:n + 2])
            cond = np.allclose(new, states[n + 1], atol=tol) if iters > 1 else 1
            states[n + 1] = new
            fvals[n + 1] = f(t[n + 1], states[n + 1])
            if cond:
                break

    return states


def ABM4(f, state0, t):
    N = len(t)
    dim = (N,) + np.shape(state0)

    # initialize state array with initial values
    states = np.zeros(dim)
    states[:4] = RungeKutta4(f, state0, t[:4])

    # initialize ODE function value array
    fvals = np.zeros(dim)
    for n in range(4):
        fvals[n] = f(t[n], states[n])

    # prediction-correction cycle
    for n in range(3, N - 1):
        h = t[n + 1] - t[n]

        # predictor: Adams-Bashforth method
        states[n + 1] = states[n] + h / 24 * (55 * fvals[n] - 59 * fvals[n - 1] + 37 * fvals[n - 2] - 9 * fvals[n - 3])

        # evaluation
        fvals[n + 1] = f(t[n + 1], states[n + 1])

        # corrector: Adams-Moulton
        states[n + 1] = states[n] + h / 720 * (
                251 * fvals[n + 1] + 646 * fvals[n] - 264 * fvals[n - 1] + 106 * fvals[n - 2] - 19 * fvals[n - 3])

        # evaluation
        fvals[n + 1] = f(t[n + 1], states[n + 1])

    return states


def LorenzSystem(sigma, r, b):
    def ODE(t, state):
        x, y, z = state
        return np.array([sigma * (y - x), x * (r - z) - y, x * y - b * z])

    return ODE


def div(states, REF, N):
    x = states.T[0]
    xREF = REF.T[0]
    for n in range(len(x)):
        if abs(x[n] - xREF[n]) > 10:
            return n / N


sigma = 10
r = 28
b = 8 / 3
T = 100
M = 2**16
Ns = 2 ** np.arange(10, 15)
state0 = (1, 1, 1)

f = LorenzSystem(sigma, r, b)

tREF = np.arange(0, T, 1 / M)
statesREF = ABM(f, state0, tREF, iters=100)

timesFE = ()
timesBE = ()
timesTR = ()
timesRK = ()
timesPC = ()

divsFE = ()
divsBE = ()
divsTR = ()
divsRK = ()
divsPC = ()

for N in Ns:
    k = M // N
    REF = statesREF[::k]
    t = np.arange(0, T, 1/N)

    start = process_time()
    statesFE = forwardEuler(f, state0, t)
    end = process_time()
    timesFE += (end - start,)

    start = process_time()
    statesBE = backwardEuler(f, state0, t)
    end = process_time()
    timesBE += (end - start,)

    start = process_time()
    statesTR = trapezoidalRule(f, state0, t)
    end = process_time()
    timesTR += (end - start,)

    start = process_time()
    statesRK = RungeKutta4(f, state0, t)
    end = process_time()
    timesRK += (end - start,)

    start = process_time()
    statesPC = ABM4(f, state0, t)
    end = process_time()
    timesPC += (end - start,)

    divsFE += (div(statesFE, REF, N),)
    divsBE += (div(statesBE, REF, N),)
    divsTR += (div(statesTR, REF, N),)
    divsRK += (div(statesRK, REF, N),)
    divsPC += (div(statesPC, REF, N),)

with open("comparison.txt", "a") as file:
    file.write(f"Comparison with M={M}\n")
    file.write(f"Ns = {Ns}\n")

    file.write(f"divsFE = {divsFE}\n")
    file.write(f"divsBE = {divsBE}\n")
    file.write(f"divsTR = {divsTR}\n")
    file.write(f"divsRK = {divsRK}\n")
    file.write(f"divsPC = {divsPC}\n")

    file.write(f"timesFE = {timesFE}\n")
    file.write(f"timesBE = {timesBE}\n")
    file.write(f"timesTR = {timesTR}\n")
    file.write(f"timesRK = {timesRK}\n")
    file.write(f"timesPC = {timesPC}\n")
