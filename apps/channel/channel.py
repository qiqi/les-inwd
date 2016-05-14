import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '../..'))

import argparse
import numpy
import les

parser = argparse.ArgumentParser()
parser.add_argument('RE', type=float)
parser.add_argument('NSTEPS', type=int)
args = parser.parse_args()

ni, nj, nk, dt = 60, 20, 40, 0.01

les.settings.lx = 6
les.settings.ly = 2
les.settings.lz = 4
les.settings.dt = dt
les.settings.mu = 1 / args.RE

def extend_u(u):
    a = 0.8
    u_ext = numpy.zeros([u.shape[0], u.shape[1]+2, u.shape[2]+2, u.shape[3]+2])
    u_ext[:,1:-1,1:-1,1:-1] = u
    u_ext[:,1:-1,1:-1,0]  = u[:,:,:,-1]
    u_ext[:,1:-1,1:-1,-1] = u[:,:,:,0]
    u_ext[:,0,:,:]  = u_ext[:,-2,:,:]
    u_ext[:,-1,:,:] = u_ext[:,1,:,:]
    u_ext[:,:,0,:]  = u_ext[:,:,1,:] * a #- u_ext[:,:,1,:].mean() * (1+a)
    u_ext[:,:,-1,:] = u_ext[:,:,-2,:] * a #- u_ext[:,:,-2,:].mean() * (1+a)
    return u_ext

def extend_p(p):
    p_ext = numpy.zeros([p.shape[0]+2, p.shape[1]+2, p.shape[2]+2])
    p_ext[1:-1,1:-1,1:-1] = p
    p_ext[1:-1,1:-1,0]  = p[:,:,-1]
    p_ext[1:-1,1:-1,-1] = p[:,:,0]
    p_ext[0,:,:]  = p_ext[-2,:,:]
    p_ext[-1,:,:] = p_ext[1,:,:]
    p_ext[:,0,:]  = p_ext[:,1,:] * 0
    p_ext[:,-1,:] = p_ext[:,-2,:] * 0
    return p_ext

source = numpy.zeros([3, ni, nj, nk])
source[0,:,:,:] = 1

state = numpy.load('initial_state.npy')
assert state.shape == (16, ni, nj, nk)
u = state[:3]
u_bar = state[3:3+2*6].reshape([2, 6, ni, nj, nk])
p = state[-1]

open('channel_quantities.txt', 'wt').close()

for i in range(args.NSTEPS):
    with open('channel.log', 'at') as f:
        u, u_bar, p = les.timestep(u, u_bar, p, source, f_log=f,
                                   extend_u=extend_u, extend_p=extend_p)
    with open('channel_quantities.txt', 'at') as f:
        f.write(('{:.18e} ' * u.shape[2] + '\n').format(
            *tuple(u[0].mean(0).mean(1))
        ))
    if i % 10 == 0:
        with open('channel_{0}.tec'.format(i), 'w') as f:
            les.tecplot_write(f, u, p)

numpy.save('final_state.npy', numpy.concatenate([
    u, u_bar.reshape([2*6,ni,nj,nk]), p.reshape([1,ni,nj,nk])]))
