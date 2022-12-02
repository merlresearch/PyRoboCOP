#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from collections import namedtuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"  # MacOS
# plt.rcParams['animation.ffmpeg_path'] = u'C:\\ffmpeg\\bin\\ffmpeg.exe' # Windows

gate = namedtuple("gate", ["radiuses", "phi"])


def rotation(beta, gamma):

    RX = np.matrix([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])

    RY = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])

    return RX, RY


class VisualizeMaze:
    def __init__(self, theta=None, radius=None, ring_radius=None, gates=None, polygon_edges=100):

        if theta is None:
            # print 'Initializing theta ...'
            self.theta = 0 * self.gamma
        else:
            self.theta = theta

        if radius is None:
            # print 'Initializing ball radius ...' # Kind of bad name!
            self.radius = 1.0 * np.ones_like(self.gamma)
        else:
            self.radius = radius

        if ring_radius is None:
            # print 'Initializing ring radius ...'
            self.ring_radius = [0.5, 1.0, 1.5]
        else:
            self.ring_radius = ring_radius

        if gates is None:
            # print 'Initializing gates ...'
            self.gates = [gate([0.5, 1.0], 0.0), gate([1.0, 1.5], pi / 2), gate([1.0, 1.5], pi)]
        else:
            self.gates = gates

        self.maze_polygon_no = polygon_edges + 1
        rho = np.linspace(0, 2 * pi, self.maze_polygon_no)
        self.x0 = np.cos(rho)
        self.y0 = np.sin(rho)

        self.circle_points = np.array([self.x0, self.y0])
        self.coord_centre = np.array([1.0, 0.0])
        self.coord_centre.shape = (2, 1)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.line = []
        for r in self.ring_radius:
            self.line.append(self.ax.plot([], [], "b-", lw=2)[0])
            self.line.append(self.ax.plot([], [], "g-", lw=2)[0])

        if len(self.gates) > 0:
            self.gate_points = []
            for g in self.gates:
                rad_pnt = np.linspace(g.radiuses[0], g.radiuses[1], 10)
                self.x_g = rad_pnt * np.cos(g.phi)
                self.y_g = rad_pnt * np.sin(g.phi)
                self.gate_points.append(np.array([self.x_g, self.y_g]))
                self.line.append(self.ax.plot([], [], "k--", lw=2)[0])

            # self.line.append(self.ax.plot([],[], 'o', color='red', ms = 10)[0])
            # self.line.append(self.ax.plot([],[], 'x', color='red', ms = 6, lw=2)[0])

        self.ball = self.ax.plot([], [], "*")

        # Set limits
        max_rad = np.max(self.ring_radius)
        self.ax.set_xlim([-max_rad * 1.1, max_rad * 1.1])
        self.ax.set_xlabel("X [cm]")
        self.ax.set_ylim([-max_rad * 1.1, max_rad * 1.1])
        self.ax.set_ylabel("Y [cm]")

        self.ani = []

    def animate(self, i):

        ind = i % len(self.theta)
        line_counter = 0
        for i, r in enumerate(self.ring_radius):
            x_new, y_new = r * self.circle_points[0], r * self.circle_points[1]
            self.line[2 * i].set_data(
                x_new[0 : int(self.maze_polygon_no / 2 + 1)], y_new[0 : int(self.maze_polygon_no / 2 + 1)]
            )
            self.line[2 * i + 1].set_data(
                x_new[int(self.maze_polygon_no / 2) :], y_new[int(self.maze_polygon_no / 2) :]
            )
            line_counter += 2

        for i in range(len(self.gates)):
            x_new, y_new = self.gate_points[i][0], self.gate_points[i][1]
            self.line[line_counter + i].set_data(x_new, y_new)

        ball = np.array([self.radius[ind] * np.cos(self.theta[ind]), self.radius[ind] * np.sin(self.theta[ind])])
        ball.shape = (2, 1)
        self.line[-2].set_data(ball[0], ball[1])
        self.line[-1].set_data(self.coord_centre[0], self.coord_centre[1])

        return self.line  # , self.ball # #, time_text

    def show_animation(self, save_filename=None):

        self.ani = animation.FuncAnimation(
            self.fig, self.animate, 800, interval=33, blit=True  # interval = delay in milliseconds between two frames
        )  # init_func=init)

        if save_filename is not None:
            mywriter = animation.FFMpegWriter(fps=15, bitrate=2000000)
            self.ani.save(save_filename + ".mp4", writer=mywriter)  # ,

    def plot_traj(self, N=10):
        # Plot the maze
        self.animate(1)
        # Plot the ball movement
        for i in range(N):
            ball = np.array([self.radius[i] * np.cos(self.theta[i]), self.radius[i] * np.sin(self.theta[i])])
            ball.shape = (2, 1)
            self.ax.plot([ball[0]], [ball[1]], "o", color="red", ms=10, fillstyle="none")[0]


if __name__ == "__main__":

    # plt.rcParams['animation.ffmpeg_path'] = u'/usr/local/bin/ffmpeg'
    #    plt.rcParams['animation.ffmpeg_path'] = u'C:\\ffmpeg\\bin\\ffmpeg.exe' # Windows

    data_points = 200
    my_gamma = 0.2 * pi * np.sin(np.linspace(0, 4 * pi, data_points))
    my_beta = 0.5 * pi * np.cos(np.linspace(0, 4 * pi, data_points))
    my_theta = np.linspace(0, 4 * pi, data_points)
    my_radius = np.ones(data_points)  # *np.abs(np.cos(my_theta))

    vis = VisualizeMaze(theta=my_theta, radius=my_radius)
    # print ('.. Showing Video ..')
    # vis.show_animation(save_filename='havij') #save_filename='havij')
    vis.plot_traj(data_points)
    plt.show()
