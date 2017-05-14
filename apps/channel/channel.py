from __future__ import division, print_function

import os
import sys
import pdb
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '../..'))

from numpy import *
from navierstokes import NavierStokes

class Channel(NavierStokes):
    def __init__(self, xgrid, ygrid, zgrid, nu, dt, tol=1E-12,
                 f_log=None, extend_u=None, extend_p=None):
        NavierStokes.__init__(self, xgrid, ygrid, zgrid, nu, dt, tol,
                f_log, extend_u=self.extend_u, extend_p=self.extend_p)
        self.last_u_exit = None
        v_upper_wall = -sin(xgrid / xgrid.max() * 2 * pi)
        self.v_upper_wall = (v_upper_wall[1:] + v_upper_wall[:-1]) / 2 * 0.5

    def extend_u(self, u):
        if self.last_u_exit is None:
            self.last_u_exit = u[:,-1].copy()

        u_ext = zeros([u.shape[0], u.shape[1]+2, u.shape[2]+2, u.shape[3]+2])
        u_ext[:,1:-1,1:-1,1:-1] = u
        u_ext[:,1:-1,1:-1,0]  = u[:,:,:,-1]
        u_ext[:,1:-1,1:-1,-1] = u[:,:,:,0]

        u_ext[:,0,:,:] = array([1,0,0])[:,newaxis,newaxis]
        inflow = (u[0,0].mean() + 1) / 2

        u_out = max(0, self.last_u_exit.mean())
        dt, dx = self.settings.dt, self.settings.dx[-1,0,0]
        relax = 1 / (1 + u_out * dt / dx)
        u_ext[:,-1,1:-1,1:-1] = relax * u[:,-1] + (1 - relax) * self.last_u_exit
        outflow = (1 + relax) / 2 * u[0,-1].mean() \
                + (1 - relax) / 2 * self.last_u_exit[0].mean()
        u_ext[0,-1] += (inflow - outflow) * 2

        u_ext[:,:,0,:]  = u_ext[:,:,1,:] * array([-1,-1,1])[:,newaxis,newaxis]
        u_ext[:,:,-1,:] = u_ext[:,:,-2,:] * array([1,-1,1])[:,newaxis,newaxis]
        u_ext[1,1:-1,-1,:]  += self.v_upper_wall[:,newaxis]
        return u_ext

    def extend_p(self, p):
        p_ext = zeros([p.shape[0]+2, p.shape[1]+2, p.shape[2]+2])
        p_ext[1:-1,1:-1,1:-1] = p
        p_ext[1:-1,1:-1,0]  = p[:,:,-1]
        p_ext[1:-1,1:-1,-1] = p[:,:,0]
        p_ext[0,:,:]  = p_ext[1,:,:]
        p_ext[-1,:,:] = p_ext[-2,:,:]
        p_ext[:,0,:]  = p_ext[:,1,:]
        p_ext[:,-1,:] = p_ext[:,-2,:]
        return p_ext

    def step(self):
        NavierStokes.step(self)
        self.last_u_exit = ns.u[:,-1].copy()

nu, Lx, Ly, Lz, T = 1E-6, 10, 1, 0.1, 10
ni, nj, nk, dt = 100, 10, 1, 0.01

open('channel_quantities.txt', 'wt').close()
f = open('channel_quantities.txt', 'wt')
ns = Channel(linspace(0, Lx, ni+1),
             linspace(0, Ly, nj+1),
             linspace(0, Lz, nk+1), nu, dt, f_log=f)

if os.path.exists('initial_state.npz'):
    ns.load('initial_state.npz')
else:
    ns.init(zeros([3,ni,nj,nk]) + array([1,0,0])[:,newaxis,newaxis,newaxis])

for i in range(int(round(T/dt))):
    ns.timestep(); f.flush()
    if i % 100 == 0:
        ns.tecwrite('{0:05d}.tec'.format(i))

ns.save('final_state.npz')
