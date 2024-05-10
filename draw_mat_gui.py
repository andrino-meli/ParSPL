#!/usr/bin/env python3
import argparse, json, os, sys, time
import scipy.sparse as spa
from enum import Enum
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

from general import *

np.random.seed(0)
MIN = 1
MAX = 7
DIV = 10

class State(Enum):
    LEFT = 0
    RIGHT = 1
    NONE = 2

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name of system.')
    parser.add_argument('dim', type=int, help='Dimension of sparse linear system matrix.', default=None)
    parser.add_argument('--edit', action='store_true',
        help='Edit an allready existing file.')

    args = parser.parse_args()
    name = args.name
    filename = f'./src/{name}.json'
    assert(len(name) > 1)
    if args.edit:
        assert(os.path.exists(filename))
    else:
        assert(not os.path.exists(filename))

    # plotting
    fig, ax = plt.subplots()
    ax.set_title(f'sparsity pattern of {name}')
    #ax.set_xlim([-.5,dim-.5])
    #ax.set_ylim([dim-.5,-.5])
    #ax.grid()

    # thread communication and state variables
    doexit = False                # if set to True exit GUI mode
    if not args.edit:
        dim = args.dim
        assert(dim > 0)
        grid = np.zeros((dim,dim))     # matrix to generate
        linsys = NameSpace()
    else:
        linsys = NameSpace.load_json(filename)
        L = spa.csc_matrix((linsys.Lx,linsys.Li,linsys.Lp),shape=(linsys.n,linsys.n))
        grid = L.toarray()
        dim = linsys.n

    state = State.NONE

    def press_handler(event):
        global state, grid
        val = None
        if event.button == 1:
            state = State.LEFT
            val = random.randint(MIN,MAX)/DIV
        elif event.button == 3:
            state = State.RIGHT
            val = 0
        if val is not None:
            x,y = event.ydata, event.xdata
            if x is None or y is None:
                return
            x,y = round(x), round(y)
            if x <= y:
                return
            grid[x,y] = val

    def release_handler(event):
        global state
        state = State.NONE

    def motion_handler(event):
        global state, grid
        if state is not State.NONE:
            x,y = event.ydata, event.xdata
            if x is None or y is None:
                return
            x,y = int(x), int(y)
            if x <= y:
                return
            elif state is State.LEFT:
                grid[x][y] = random.randint(MIN,MAX)/DIV
            elif state is State.RIGHT:
                grid[x][y] = 0

    def key_handler(event):
        global doexit
        if (event.key == 'ctrl+d' or event.key == 'x' or event.key == 'escape'):
            doexit = True
            print('exiting ...')

    fig.canvas.mpl_connect('button_press_event', press_handler)
    fig.canvas.mpl_connect('button_release_event', release_handler)
    fig.canvas.mpl_connect('motion_notify_event', motion_handler)
    fig.canvas.mpl_connect('key_press_event', key_handler)
    # run plotting in this thread

    ax.plot((-0.5,dim),(-0.5,dim),'k')
    img = ax.imshow(grid, cmap='gist_gray_r', interpolation='none',vmin=0/DIV,vmax=MAX/DIV)
    plt.pause(0.01)
    plt.draw()
    while not doexit:
        img.set_data(grid)
        plt.pause(0.0001)
    plt.close()

    # Save data
    L = spa.csc_matrix(grid)
    linsys.n = dim
    linsys.Lp = L.indptr
    linsys.Li = L.indices
    linsys.Lx = L.data

    # two elemenet Diagonal
    linsys.D = 2*np.ones(dim)
    #linsys.D = np.random.lognormal(1e3,1e3,dim)
    #assert(not np.any(linsys.D == 0.0))

    # compute K = LDLT
    L = spa.csc_matrix((linsys.Lx,linsys.Li,linsys.Lp),shape=(dim,dim))
    D = spa.diags(linsys.D,0)
    K = L @ D @ L.transpose()
    linsys.Kp = K.indptr
    linsys.Ki = K.indices
    linsys.Kx = K.data

    bprint(f'Dumping data to {filename}')
    linsys.dump_json(filename)

