import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

import argparse
import numpy
import les

parser = argparse.ArgumentParser()
parser.add_argument('RE', type=float)
parser.add_argument('U0', type=float)
parser.add_argument('NSTEPS', type=int)
args = parser.parse_args()

ni, nj, nk, dt = 80, 20, 20, 0.25

les.settings.lx = 8
les.settings.ly = 2
les.settings.lz = 2
les.settings.dt = dt
les.settings.mu = 1 / args.RE

fan = numpy.cos(numpy.pi * numpy.arange(ni) / ni)**128
fan = fan[numpy.newaxis,:,numpy.newaxis,numpy.newaxis]
u_inlet = numpy.sin(numpy.pi * numpy.arange(nj) / nj)**16 * args.U0
u_inlet = numpy.outer([1,0,0], u_inlet)[:,numpy.newaxis,:,numpy.newaxis]

state = numpy.load('initial_state.npy')
assert state.shape == (16, ni, nj, nk)
u = state[:3]
u_bar = state[3:3+2*6].reshape([2, 6, ni, nj, nk])
p = state[-1]

open('jet_quantities.txt', 'wt').close()

for i in range(args.NSTEPS):
    source = fan * (u_inlet - u) * 2
    u, u_bar, p = les.timestep(u, u_bar, p, source)
    with open('jet_quantities.txt', 'at') as f:
        f.write('{0:.18e} {1:.18e} \n'.format(
            les.kinetic_energy(u),
            u[1,ni/2,0,0]
        ))
    if i % 10 == 0:
        with open('jet_{0}.tec'.format(i), 'w') as f:
            les.tecplot_write(f, u, p)

numpy.save('final_state.npy', numpy.concatenate([
    u, u_bar.reshape([2*6,ni,nj,nk]), p.reshape([1,ni,nj,nk])]))
