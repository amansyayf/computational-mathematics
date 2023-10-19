import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from scipy import special

N = 9

h = 0.1
L = N * h
tau = 0.1

lambda_= 0.1*h**2/tau
T0 = 2
sigma = 1
c = np.sqrt(lambda_*T0**sigma/sigma)
epsilon = 1e-2
teta = 0.5

# scheme
def true_func1(x, t):
    return T0*special.erfc(x/(2*np.sqrt(lambda_*t)))

def zero_scheme1(Tmp1, Tm, Tmm1):
    return lambda_*tau*(Tmp1-2*Tm+Tmm1)/h**2+Tm

def true_func2(x, t):
    return (sigma*c*(c*t-x)/lambda_)**(1/sigma) if x < c*t else 0

def zero_scheme2(Tmp1, Tm, Tmm1):
    lambdap = 2*lambda_*(Tm*Tmp1)**(sigma)/(Tm**sigma+Tmp1**sigma)
    lambdam = 2*lambda_*(Tm*Tmm1)**(sigma)/(Tm**sigma+Tmm1**sigma)
    return tau*(lambdap*(Tmp1-Tm) - lambdam*(Tm-Tmm1))/h**2+Tm

def backward(Tmp1, Pmp1, Qmp1):
    return Pmp1*Tmp1 + Qmp1
def forward3(Pm, Qm, Tmp1, Tm, Tmm1):
    bm = 2+h**2/tau/teta/lambda_
    dm = -(1-teta)/teta*(Tmp1-2*Tm+Tmm1)-Tm*h**2/tau/teta/lambda_
    return 1/(bm - Pm), (Qm -dm)/(bm - Pm)


def forward4(Pm, Qm, Tmp1, Tm, Tmm1):
    lambdap = 2 * lambda_ * (Tm * Tmp1) ** (sigma) / (Tm ** sigma + Tmp1 ** sigma)
    lambdam = 2 * lambda_ * (Tm * Tmm1) ** (sigma) / (Tm ** sigma + Tmm1 ** sigma)
    am = lambdam
    bm = lambdap + lambdam + h ** 2 / tau / teta
    cm = lambdap
    dm = -(1 - teta) / teta * (lambdap * (Tmp1 - Tm) - lambdam * (Tm - Tmm1)) - Tm * h ** 2 / tau / teta

    return cm / (bm - am * Pm), (am * Qm - dm) / (bm - am * Pm)

def forward5(Pm, Qm, Tm, prevTm):
    bm = 2+h**2*(1+teta)/tau/lambda_
    dm =  -teta*h**2*(Tm-prevTm)/tau/lambda_-(1+teta)*h**2*Tm/tau/lambda_
    return 1/(bm - Pm), (Qm -dm)/(bm - Pm)


def forward6(Pm, Qm, Tmp1, Tm, Tmm1, prevTm):
    lambdap = 2 * lambda_ * (Tm * Tmp1) ** (sigma) / (Tm ** sigma + Tmp1 ** sigma)
    lambdam = 2 * lambda_ * (Tm * Tmm1) ** (sigma) / (Tm ** sigma + Tmm1 ** sigma)
    am = lambdam
    bm = lambdap + lambdam + (1 + teta) * h ** 2 / tau
    cm = lambdap
    dm = -teta * h ** 2 * (Tm - prevTm) / tau - (1 + teta) * h ** 2 * Tm / tau

    return cm / (bm - am * Pm), (am * Qm - dm) / (bm - am * Pm)


class UpdatePlot:
    def __init__(self, ax, title='', flag = None):
        self.flag = flag

        self.notes = np.arange(0, L, h)
        self.spec_notes = np.arange(0, L, h / 3)

        if self.flag in [1, 3, 5]:
            self.true_u = np.zeros(3 * N)
            self.prev_u = np.zeros(N)
            self.next_u = np.zeros(N)
            self.true_u[0] = T0
            self.prev_u[0] = T0
            self.next_u[0] = T0
            if self.flag in [3, 5]:
                self.P = np.zeros(N - 1)
                self.Q = np.ones(N - 1) * T0
                if self.flag == 5:
                    self.prevprev_u = np.zeros(N)
                    self.prevprev_u[0] = T0

        elif self.flag in [2, 4, 6]:
            self.true_u = np.zeros(3 * N)
            self.prev_u = np.ones(N) * epsilon
            self.next_u = np.ones(N) * epsilon
            self.prev_u[0] = 0
            self.next_u[0] = T0 * tau ** (1 / sigma)
            if self.flag in [4, 6]:
                self.P = np.zeros(N - 1)
                self.Q = np.ones(N - 1) * T0 * tau ** (1 / sigma)
                if self.flag == 6:
                    self.prevprev_u = np.ones(N) * epsilon

        self.line, = ax.plot([], [], c='black')
        color = random.choice(['blue', 'green', 'purple', 'olive', 'darkorchid', 'deepskyblue', 'darkgoldenrod'])
        self.scat = ax.scatter([], [], c=color)
        self.ax = ax
        if self.flag in [1, 3, 5]:
            self.ax.set_ylim(-0.25, 2.25)
        else:
            self.ax.set_ylim(-0.25, 13)
        self.ax.set_xlim(-0.01, L)
        self.ax.set_title(title)
        self.ax.grid(True)


    def __call__(self, j):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if j == 0:
            self.line.set_data(self.spec_notes, self.true_u)
            self.scat.set_offsets(np.stack([self.notes, self.prev_u]).T)
            return self.line, self.scat

        if self.flag in [1, 3, 5]:
            self.true_u = np.array(list(map(lambda x: true_func1(x, (j + 1) * tau), self.spec_notes)))
            if self.flag == 1:
                for i in range(1, N - 1):
                    self.next_u[i] = zero_scheme1(self.prev_u[i + 1], self.prev_u[i], self.prev_u[i - 1])
            elif self.flag in [3, 5]:
                if self.flag == 3:
                    for i in range(1, N - 1):
                        self.P[i], self.Q[i] = forward3(self.P[i - 1], self.Q[i - 1], self.prev_u[i + 1],
                                                         self.prev_u[i], self.prev_u[i - 1])
                elif self.flag == 5:
                    if j == 1:
                        for i in range(1, N - 1):
                            self.P[i], self.Q[i] = forward3(self.P[i - 1], self.Q[i - 1], self.prev_u[i + 1],
                                                             self.prev_u[i], self.prev_u[i - 1])
                    else:
                        for i in range(1, N - 1):
                            self.P[i], self.Q[i] = forward5(self.P[i - 1], self.Q[i - 1], self.prev_u[i],
                                                             self.prevprev_u[i])
                        self.prevprev_u = self.prev_u
                for i in range(N - 2, -1, -1):
                    self.next_u[i] = backward(self.prev_u[i + 1], self.P[i], self.Q[i])
            self.prev_u = self.next_u
            self.next_u = np.zeros(N)
            self.next_u[0] = T0

        elif self.flag in [2, 4, 6]:
            self.true_u = np.array(list(map(lambda x: true_func2(x, (j + 1) * tau), self.spec_notes)))
            if self.flag == 2:
                for i in range(1, N - 1):
                    self.next_u[i] = zero_scheme2(self.prev_u[i + 1], self.prev_u[i], self.prev_u[i - 1])
            elif self.flag in [4, 6]:
                if self.flag == 4:
                    for i in range(1, N - 1):
                        self.P[i], self.Q[i] = forward4(self.P[i - 1], self.Q[i - 1], self.prev_u[i + 1],
                                                        self.prev_u[i], self.prev_u[i - 1])
                elif self.flag == 6:
                    if j == 0:
                        for i in range(1, N - 1):
                            self.P[i], self.Q[i] = forward4(self.P[i - 1], self.Q[i - 1], self.prev_u[i + 1],
                                                            self.prev_u[i], self.prev_u[i - 1])
                    else:
                        for i in range(1, N - 1):
                            self.P[i], self.Q[i] = forward6(self.P[i - 1], self.Q[i - 1], self.prev_u[i + 1],
                                                            self.prev_u[i], self.prev_u[i - 1], self.prevprev_u[i])
                        self.prevprev_u = self.prev_u
                for i in range(N - 2, -1, -1):
                    self.next_u[i] = backward(self.prev_u[i + 1], self.P[i], self.Q[i])
                self.Q[0] = T0 * ((j + 2) * tau) ** (1 / sigma)
            self.prev_u = self.next_u
            self.next_u = np.ones(N) * epsilon
            self.next_u[0] = T0 * ((j + 2) * tau) ** (1 / sigma)

        self.line.set_data(self.spec_notes, self.true_u)
        self.scat.set_offsets(np.stack([self.notes, self.prev_u]).T)

        return self.line, self.scat


def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ud = UpdatePlot(ax, title='0 scheme, 1 equation ', flag=1)
    # ud = UpdatePlot(ax, title='0 scheme, 4 equation', flag=2)
    # ud = UpdatePlot(ax, title='1 scheme, 1 equation', flag=3)
    # ud = UpdatePlot(ax, title='1 scheme, 4 equation', flag=4)
    # ud = UpdatePlot(ax, title='3 scheme, 1 equation ', flag=5)
    # ud = UpdatePlot(ax, title='3 scheme, 4 equation ', flag=6)
    anim = FuncAnimation(fig, ud, frames=60, interval=55, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':
    main()