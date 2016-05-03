import numpy
import les

ni, nj, nk, dt = 200, 40, 20, 0.1

les.settings.lx = 10
les.settings.ly = 2
les.settings.lz = 1
les.settings.dt = dt
les.settings.mu = 0.0001

u_inlet = numpy.sin(numpy.pi * numpy.arange(nj) / nj)**12
u_inlet = u_inlet[numpy.newaxis,:,numpy.newaxis]

u = numpy.random.random([3, ni, nj, nk])
u_bar = numpy.random.random([2, 6, ni, nj, nk]) * 0.1
p = numpy.random.random([ni, nj, nk])

for i in range(1001):
    u, u_bar, p = les.timestep(u, u_bar, p)
    u[0,:nj/2,:] += 0.5 * (u_inlet - u[0,:nj/2,:])
    u[1:3,:nj/2,:] *= 0.5
    if i % 10 == 0:
        with open('shear_{0}.tec'.format(i), 'w') as f:
            les.tecplot_write(f, u, p)
