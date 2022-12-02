#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from collections import namedtuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
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
    def __init__(
        self, beta=None, gamma=None, theta=None, radius=None, ring_radius=None, gates=None, polygon_edges=100
    ):

        if gamma is None:
            # print 'Initializing gamma ...'
            self.gamma = np.linspace(0, 2 * pi, 50)
        else:
            self.gamma = gamma

        if beta is None:
            # print 'Initializing beta ...'
            self.beta = 0 * self.beta
        else:
            self.beta = beta

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

        self.maze_polygon_no = polygon_edges  # + 1
        rho = np.linspace(0, 2 * pi, self.maze_polygon_no)
        self.x0 = np.cos(rho)
        self.y0 = np.sin(rho)
        self.z0 = 0 * np.ones_like(self.x0)

        self.circle_points = np.array([self.x0, self.y0, self.z0])

        self.coord_centre = np.array([1.0, 0.0, 0.0])
        self.coord_centre.shape = (3, 1)

        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)

        self.ax.view_init(elev=35.264, azim=45.0)

        self.line = []
        for r in self.ring_radius:
            self.line.append(self.ax.plot([], [], [], "b-", lw=2)[0])
            self.line.append(self.ax.plot([], [], [], "g-", lw=2)[0])

        if len(self.gates) > 0:
            self.gate_points = []

            for g in self.gates:

                rad_pnt = np.linspace(g.radiuses[0], g.radiuses[1], 10)
                self.x_g = rad_pnt * np.cos(g.phi)
                self.y_g = rad_pnt * np.sin(g.phi)
                self.z_g = 0 * np.ones_like(self.x_g)

                self.gate_points.append(np.array([self.x_g, self.y_g, self.z_g]))

                self.line.append(self.ax.plot([], [], [], "k--", lw=2)[0])

            self.line.append(self.ax.plot([], [], [], "o", color="orange", ms=10)[0])
            self.line.append(self.ax.plot([], [], [], "x", color="red", ms=6, lw=2)[0])

        #        print self.line

        #        self.line = [self.ax.plot([], [],[], '-', lw=2)[0],
        #                     self.ax.plot([], [],[], 'g-', lw=2)[0],
        #                     self.ax.plot([],[],[], 'o', color='orange', ms = 10)[0],
        #                     self.ax.plot([],[],[], 'x', color='red', ms = 10, lw=4)[0],
        #                     ]

        #        print self.line

        self.ball = self.ax.plot([], [], [], "o")

        # Set limits

        max_rad = np.max(self.ring_radius)

        self.ax.set_xlim3d([-max_rad, max_rad])
        self.ax.set_xlabel("X")

        self.ax.set_ylim3d([-max_rad, max_rad])
        self.ax.set_ylabel("Y")

        self.ax.set_zlim3d([-max_rad, max_rad])
        self.ax.set_zlabel("Z")

        self.ani = []

    def animate(self, i):

        #        gamma = pi/100*i

        ind = i % len(self.gamma)

        RX, RY = rotation(self.beta[ind], self.gamma[ind])

        new_points_unit = np.array(RY * RX * self.circle_points)

        line_counter = 0
        for i, r in enumerate(self.ring_radius):
            #            print r

            x_new, y_new, z_new = r * new_points_unit[0], r * new_points_unit[1], r * new_points_unit[2]

            self.line[2 * i].set_data(
                x_new[0 : int(self.maze_polygon_no / 2 + 1)], y_new[0 : int(self.maze_polygon_no / 2 + 1)]
            )
            self.line[2 * i].set_3d_properties(z_new[0 : int(self.maze_polygon_no / 2 + 1)])

            self.line[2 * i + 1].set_data(
                x_new[int(self.maze_polygon_no / 2) :], y_new[int(self.maze_polygon_no / 2) :]
            )
            self.line[2 * i + 1].set_3d_properties(z_new[int(self.maze_polygon_no / 2) :])

            line_counter += 2
        #            print i

        #        print 'Here!'

        for i in range(len(self.gates)):

            new_gate_points = np.array(RY * RX * self.gate_points[i])

            x_new, y_new, z_new = new_gate_points[0], new_gate_points[1], new_gate_points[2]
            #            print x_new

            self.line[line_counter + i].set_data(x_new, y_new)
            self.line[line_counter + i].set_3d_properties(z_new)

        #        new_points = np.array(RY*RX*self.circle_points)
        #
        #        x_new, y_new, z_new = new_points[0],new_points[1],new_points[2]
        #
        #        self.line[0].set_data(  x_new[0:self.maze_polygon_no/2+1],
        #                            y_new[0:self.maze_polygon_no/2+1])
        #        self.line[0].set_3d_properties(z_new[0:self.maze_polygon_no/2+1])
        #
        #        self.line[1].set_data(  x_new[self.maze_polygon_no/2:], y_new[self.maze_polygon_no/2:])
        #        self.line[1].set_3d_properties(z_new[self.maze_polygon_no/2:])

        ball = np.array([self.radius[ind] * np.cos(self.theta[ind]), self.radius[ind] * np.sin(self.theta[ind]), 0.0])
        ball.shape = (3, 1)

        ball_new = RY * RX * ball
        #        print ball_new

        self.line[-2].set_data(ball_new[0], ball_new[1])
        self.line[-2].set_3d_properties(ball_new[2])

        #        self.line[2].set_data(ball_new[0],ball_new[1])
        #        self.line[2].set_3d_properties(ball_new[2])

        coord_centre_new = RY * RX * self.coord_centre
        self.line[-1].set_data(coord_centre_new[0], coord_centre_new[1])
        self.line[-1].set_3d_properties(coord_centre_new[2])

        #        self.line[3].set_data(coord_centre_new[0],coord_centre_new[1])
        #        self.line[3].set_3d_properties(coord_centre_new[2])

        #        self.ball.set_data(1.,1.)
        #        self.ball.set_3d_properties(1.)

        #        print self.line

        return self.line  # , self.ball # #, time_text

    def show_animation(self, save_filename=None):

        self.ani = animation.FuncAnimation(
            self.fig, self.animate, 800, interval=33, blit=True  # interval = delay in milliseconds between two frames
        )  # init_func=init)

        if save_filename is not None:
            mywriter = animation.FFMpegWriter(fps=15, bitrate=2000000)
            self.ani.save(save_filename + ".mp4", writer=mywriter)  # ,


#                      extra_args=['-vcodec', 'libx264', '-x264opts','qp 0'])
#        self.ani.save('maze.mp4', fps=15)

#        plt.show()


if __name__ == "__main__":

    plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"
    #    plt.rcParams['animation.ffmpeg_path'] = u'C:\\ffmpeg\\bin\\ffmpeg.exe' # Windows

    data_points = 200
    my_gamma = 0.2 * pi * np.sin(np.linspace(0, 4 * pi, data_points))
    my_beta = 0.5 * pi * np.cos(np.linspace(0, 4 * pi, data_points))
    my_theta = np.linspace(0, 4 * pi, data_points)
    my_radius = np.ones(data_points)  # *np.abs(np.cos(my_theta))

    vis = VisualizeMaze(beta=my_beta, gamma=my_gamma, theta=my_theta, radius=my_radius)
    print(".. Showing Video ..")
    vis.show_animation(save_filename="havij")  # save_filename='havij')
