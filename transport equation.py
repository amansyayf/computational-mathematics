import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation

N = 100
n = 50
h = 0.1
L = N * h
tau = 0.1
T = n * tau
a = 0.8
u1 = 10
u2 = 0
sigma = a * tau / h

q = 0.25
epsilon = 1e-10


def true_func(x, t):
    return u1 if x < a * t else u2


def explicit_left_corner(_, __, um, um_1, ___):
    return um - sigma * (um - um_1)


def implicit_left_corner(um_1, _, um, __, ___):
    return (um + sigma * um_1) / (1 + sigma)


def reversed_implicit_left_corner(unext, uprev):
    sigma = -1.2 * tau / h
    return (1 + sigma) * unext / sigma - uprev / sigma


def laks(_, ump1, __, um_1, ___):
    return 0.5 * (ump1 + um_1) - 0.5 * sigma * (ump1 - um_1)


def laks_vendrof(_, ump1, um, um_1, __):
    return um - sigma * (um - um_1) - 0.5 * sigma * (1 - sigma) * (ump1 + um_1 - 2 * um)


def smooth_laks_vendrof(ump2, ump1, um, um_1, um_2):
    Dmm = um_1 - um_2
    Dm = um - um_1
    Dp = ump1 - um
    Dpp = ump2 - ump1
    Qp = Dp if Dpp * Dp < 0 or Dp * Dm < 0 else 0
    Qm = Dm if Dmm * Dm < 0 or Dp * Dm < 0 else 0
    return um + q * (Qp - Qm)


def beam_uorming(_, __, um, um_1, um_2):
    return um - sigma * (1.5 * um - 2 * um_1 + 0.5 * um_2) + 0.5 * sigma ** 2 * (um - 2 * um_1 + um_2)


def TVD(_, ump1, um, um_1, um_2):
    rm = (um - um_1 + epsilon) / (ump1 - um + epsilon)
    rm_1 = (um_1 - um_2 + epsilon) / (um - um_1 + epsilon)
    phi = min(2, rm) if rm > 1 else min(2 * rm, 1) if rm > 0 else 0
    phi_1 = min(2, rm_1) if rm_1 > 1 else min(2 * rm_1, 1) if rm_1 > 0 else 0
    return um - sigma * (um - um_1) - 0.5 * sigma * (1 - sigma) * (phi * (ump1 - um) - phi_1 * (um - um_1))


class UpdatePlot:
    def __init__(self, ax, title='', right_func=None, main_func=None, left_func=None, smooth_lv=False,
                 reversed_left_implicit=False):
        self.right_func = right_func
        self.main_func = main_func
        self.left_func = left_func
        self.smooth_lv = smooth_lv
        self.reversed_left_implicit = reversed_left_implicit

        self.notes = np.arange(0, L, h)
        self.spec_notes = np.arange(0, L, h / 3)
        self.true_u = np.ones(3 * N) * u2
        if not self.reversed_left_implicit:
            self.true_u[0] = u1
        else:
            self.true_u[-1] = u1

        self.line, = ax.plot([], [], c='black')
        color = random.choice(['blue', 'green', 'purple', 'olive', 'darkorchid', 'deepskyblue', 'darkgoldenrod'])
        self.scat = ax.scatter([], [], c=color)
        self.ax = ax

        self.ax.set_ylim(u2 - 2, u1 + 2)
        self.ax.set_xlim(0, L + h)
        self.ax.set_title(title)
        self.ax.grid(True)

        self.prev_u = np.ones(N) * u2
        if not self.reversed_left_implicit:
            self.prev_u[0] = u1
        else:
            self.prev_u[-1] = u1
        self.next_u = np.ones(N) * u1
        if self.smooth_lv:
            self.smooth_u = self.prev_u

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            self.line.set_data(self.spec_notes, self.true_u)
            if not self.smooth_lv:
                self.scat.set_offsets(np.stack([self.notes, self.prev_u]).T)
            else:
                self.scat.set_offsets(np.stack([self.notes, self.smooth_u]).T)
            return self.line, self.scat

        if not self.reversed_left_implicit:
            self.true_u = np.array(list(map(lambda x: true_func(x, (i + 1) * tau), self.spec_notes)))
            if self.right_func is not None:
                self.next_u[1] = self.right_func(self.next_u[0], self.prev_u[2], self.prev_u[1], self.prev_u[0], None)
            else:
                self.next_u[1] = self.main_func(self.next_u[0], self.prev_u[2], self.prev_u[1], self.prev_u[0], None)

            for j in range(2, N - 1):
                self.next_u[j] = self.main_func(self.next_u[j - 1], self.prev_u[j + 1], self.prev_u[j],
                                                self.prev_u[j - 1],
                                                self.prev_u[j - 2])

            if self.left_func is not None:
                self.next_u[-1] = self.left_func(self.next_u[-2], None, self.prev_u[-1], self.prev_u[-2],
                                                 self.prev_u[-3])
            else:
                self.next_u[-1] = self.main_func(self.next_u[-2], None, self.prev_u[-1], self.prev_u[-2],
                                                 self.prev_u[-3])

            if self.smooth_lv:
                self.smooth_u = self.next_u.copy()
                for j in range(2, N - 2):
                    self.smooth_u[j] = smooth_laks_vendrof(self.next_u[j + 2], self.next_u[j + 1], self.next_u[j],
                                                           self.next_u[j - 1], self.next_u[j - 2])


        else:
            self.true_u = np.array(list(map(lambda x: u2 if x < L - 1.2 * (i + 1) * tau else u1, self.spec_notes)))

            for j in range(N - 2, -1, -1):
                self.next_u[j] = reversed_implicit_left_corner(self.next_u[j + 1], self.prev_u[j + 1])

        self.prev_u = self.next_u
        self.next_u = np.ones(N) * u1

        self.line.set_data(self.spec_notes, self.true_u)
        if not self.smooth_lv:
            self.scat.set_offsets(np.stack([self.notes, self.prev_u]).T)
        else:
            self.scat.set_offsets(np.stack([self.notes, self.smooth_u]).T)

        return self.line, self.scat


def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ud = UpdatePlot(ax, title='Explicit left corner', main_func=explicit_left_corner)
    # ud = UpdatePlot(ax, title='Implicit left corner', main_func=implicit_left_corner)
    # ud = UpdatePlot(ax, title='Implicit left corner', reversed_left_implicit=True)
    # ud = UpdatePlot(ax, title='Laks', main_func=laks, left_func=explicit_left_corner)
    # ud = UpdatePlot(ax, title='Laks Vendrof', main_func=laks_vendrof, left_func=explicit_left_corner, smooth_lv=True)
    # ud = UpdatePlot(ax, title='Beam Uorming', main_func=beam_uorming, right_func=explicit_left_corner)
    # ud = UpdatePlot(ax, title='TVD', main_func=TVD, left_func=explicit_left_corner, right_func=explicit_left_corner)
    anim = FuncAnimation(fig, ud, frames=55, interval=200, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':
    main()
