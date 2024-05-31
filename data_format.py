import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
import math
from general import svdinvert, wprint, DEBUG, eprint, HARTS, dprint, color_palette, list2array
from bfloat16 import bfloat16


class Tile:
    ''' Abstract class to contain a tile. Inherit this by a tile class that is also a cointainer for data.'''
    def __init__(self,rowa,rowz,cola,colz):
        self.rowa = rowa # start row
        self.rowz = rowz # end row
        self.cola = cola # start col
        self.colz = colz # end col

    REDUCES = False # Does not reduce


    def empty_copy(self):
        # call super class __init__
        return type(self)(self.rowa,self.rowz,self.cola,self.colz)

    def is_diag(self):
        return self.rowa==self.cola and self.rowz==self.colz

    def is_rect(self):
        return not self.is_diag()

    def assigned_data(self):
        return self.nnz()
    
    def snum_fe(self):
        ''' Number of synchronization steps in the forward elimination.
        Some kernels have an intrinsic need for aditional synchronization steps.
        These want to increase this number accordingly.
        '''
        return 0

    def snum_bs(self):
        ''' Number of synchronization steps in the backward substitution.
        Some kernels have an intrinsic need for aditional synchronization steps.
        These want to increase this number accordingly.
        '''
        return 0

    def density(self):
        if self.is_diag:
            # The upper part is always empty so do not count it.
            l = self.rowz-self.rowa
            underdiag = l*(l-1)/2
            return self.nnz()/underdiag
        else:
            return self.nnz()/((self.rowz-self.rowa)*(self.colz-self.cola))

    def classname(self):
        classname = self.__class__.__name__
        return classname.split('.')[-1]

    def __repr__(self):
        return f'{self.classname()}_{self.rowa}r{self.rowz}_{self.cola}c{self.colz}'

    def show_on_plot(self,ax,number=None):
        plotobjs = []
        if self.is_rect():
            color = 'b'
            x = [self.cola-0.5,self.colz-0.5,self.colz-0.5,self.cola-0.5,self.cola-0.5]
            y = [self.rowa-0.5,self.rowa-0.5,self.rowz-0.5,self.rowz-0.5,self.rowa-0.5]
        else:
            color = 'lime'
            x = [self.cola-0.5,self.colz-0.5,self.cola-0.5,self.cola-0.5]
            y = [self.rowa-0.5,self.rowz-0.5,self.rowz-0.5,self.rowa-0.5]
        if self.is_diag():
            color = 'lime'
        lines = ax.plot(x,y,color=color,linewidth=1.5)
        plotobjs.extend(lines)
        if number is not None:
            # create label
            if self.snum_fe() == 0:
                label = f' s{number}\n{self.classname()}'
            else:
                label = f's{str(number)}'
                for sy in range(self.snum_fe()):
                    label += f',s{str(sy+1+number)}'
                label += f'\n{self.classname()}'
            # position label
            if self.is_diag():
                xpos = -0.5 + self.colz
                ypos = -0.5 + self.rowa
                txt = ax.text(xpos,ypos,label,fontsize=15,color=color,ha='left',va='bottom')
            else:
                xpos = -0.5 + self.cola + 0.5*(self.colz-self.cola)
                ypos = -0.5 + self.rowa + 0.5*(self.rowz-self.rowa)
                txt = ax.text(xpos,ypos,label,fontsize=15,color=color,ha='center',va='center')
            plotobjs.append(txt)
        return plotobjs

class Collist(Tile,dict):
    # dictionary containting data:
    #  key is the column number
    #  value is a tuple of the two lists: ([Li],[Lx])
    #  empty columns are forbiden

    REDUCES = True

    def nnz(self):
        nnz = 0
        for k in self:
            Li = self[k][0]
            nnz += len(Li)
        return nnz

    def insert(self,row,col,val):
        assert(val is not None)
        #print(f'Inserting {row},{col} at {self}')
        assert(self.rowa <= row < self.rowz)
        assert(self.cola <= col < self.colz)
        if col not in self:
            self[col] = ([row],[val])
        else:
            (li,lx) = self[col]
            li.append(row)
            lx.append(val)

    def empty_copy(self):
        # call super class __init__
        new = type(self)(self.rowa,self.rowz,self.cola,self.colz)
        new.reduce_stop = self.reduce_stop
        new.reductiona = self.reductiona
        new.reductionlen = self.reductionlen
        return new

    def empty(self):
        return len(self) == 0

    def merge(self,other):
        if not isinstance(other,Collist):
            raise NotImplementedError(f'Trying to merge {type(self)} with {type(other)}')
        # TODO: assert we are directly below each other:
        #       currently we assume the merging is done correctly according to a dep. tree.
        #    raise ValueError(f'{self} and {other} must be in line')
        self.rowa = min(self.rowa,other.rowa)
        self.cola = min(self.cola,other.cola)
        self.rowz = max(self.rowz,other.rowz)
        self.colz = max(self.colz,other.colz)
        for k in other:
            if k not in self:
                self[k] = other[k]
            else:
               (iO,xO) = other[k]
               Li,Lx = self[k]
               # TODO: potentially sort
               Li.extend(iO)
               Lx.extend(xO)
               #self[k] = (Li,Lx)

    def color_dict(self, sq_dict, color):
        ''' Color sq_dict according to schedule '''
        for c,(Li,Lx) in self.items():
            for r in Li:
                sq_dict[r][c].set_color(color)
        return []

    def schedule(self,cores=range(HARTS)):
        # each worker has a list of columns
        dist = [self.empty_copy() for h in cores]
        load = np.zeros(len(cores))
        # sort columns by length. The work is a Collist data structure.
        # It has the structure: {col_num: (Li,Lx)}
        sorted_cols = sorted(self.items(), key=lambda item: len(item[1][0]),reverse=True)
        # assign next longest column to least busy core.
        for col,(Li,Lx) in sorted_cols:
            least_busy_worker = np.argmin(load)
            # TODO: have more complex performance function to balance scheduling
            load[least_busy_worker] += len(Li)
            dist[least_busy_worker][col] = (Li,Lx)
        return dist

    def schedule_reduction(self,cores=range(HARTS)):
        # determine length
        start = self.rowa
        stop = self.reduce_stop
        l = (stop-start)//len(cores)
        thr = (stop-start)%len(cores)
        self.reductionlen = [l+1 if h < thr else l for h in cores]

        # determine start
        ra = [self.rowa]
        for h in cores[:-1]:
            ra.append(ra[-1]+self.reductionlen[h])
        self.reductiona = ra

    def codegen(self,s,h):
        # define names of variables and functions
        cols = f'{self}_h{h}_assigned_cols'
        len_cols = f'{self}_h{h}_len_cols'
        ri = f'{self}_h{h}_ri'
        rx = f'{self}_h{h}_rx'
        argstruct = f'{self}_h{h}_args'

        # collect all data together
        cols_dat = []
        len_cols_dat = []
        ri_dat = []
        rx_dat = []
        for k,(i,x) in self.items():
            cols_dat.append(k)
            len_cols_dat.append(len(i))
            ri_dat.extend(list(i))
            rx_dat.extend(list(x))

        # add data
        dat = {}
        dat[cols] = list2array(cols_dat,cols,base=16)
        dat[len_cols] = list2array(len_cols_dat,len_cols,base=16)
        dat[ri] = list2array(ri_dat,ri,base=16)
        dat[rx] = np.array(rx_dat)
        if self.assigned_data() == 0:
            # in case the core process no data we only support the reduction effort
            args = f'NULL, NULL, NULL, NULL, NULL, 0, 0, {self.reductiona[h]}, {self.reductionlen[h]}' 
        else:
            args = f'bp_tmp{h}, {cols}, {len_cols}, {ri}, {rx}, {len(cols_dat)}, {len(ri_dat)}, {self.reductiona[h]}, {self.reductionlen[h]}' 
        dat[argstruct] = f'Collist {argstruct} = '+'{'+ args + '};\n'

        lsolve = f'collist_lsolve(&{argstruct})'
        if self.assigned_data() == 0:
            ltsolve = f'//empty'
        else:
            ltsolve = f'collist_ltsolve(&{argstruct})'
        return (lsolve,ltsolve),dat

    def snum_fe(self):
        return 1


class Empty(Tile):

    def schedule(self,cores=range(HARTS)):
        dist = [Empty() for h in cores]
        return dist

    def codegen(self,s,h):
        return (('// empty', '// empty'),{})

    def color_dict(self,sq_dict,color):
        return []

    def show_on_plot(self,ax,number=None):
        return []

    def nnz(self):
        return 0

    def empty(self):
        return True

    def __repr__(self):
        return f'{self.classname()}'


class SynchBuffer(Empty):

    def codegen(self,s,h):
        return (('// synchronization buffer', '// synchronization buffer'),{})

    def schedule(self,cores=range(HARTS)):
        dist = [SynchBuffer() for h in cores]
        return dist


class DiagInv(Tile):
    def __init__(self,collist,empty=False):
        assert(isinstance(collist,Collist))
        self.rowa = collist.rowa
        self.rowz = collist.rowz
        self.cola = collist.cola
        self.colz = collist.colz

        # rows assigned to this instance (offset by -collist.rowa)
        self.assigned_rows = []

        # get dimensions
        self.n = self.rowz-self.rowa
        self.offset = self.rowa
        assert(self.is_diag())
        assert(self.n > 1) # 1x1 lower triag is just empty

        # fill matrix with data
        if not empty:
            self.assigned_rows = list(range(self.n,1))
            #
            self.dense_triag = np.eye(self.n)
            for col,(Li,Lx) in collist.items():
                for i,x in zip(Li,Lx):
                    c,r = col-self.offset, i - self.offset
                    assert(c < r)
                    self.dense_triag[r][c] = x

            # invert
            self.dense_inverse = np.linalg.inv(self.dense_triag)
            # check numerical stability
            self.check_numerical_stability()

    def empty_copy(self):
        tmp = DiagInv(Collist(self.rowa,self.rowz,self.cola,self.colz),empty=True)
        # copy pointers to data
        tmp.dense_triag = self.dense_triag
        tmp.dense_inverse = self.dense_inverse
        return tmp

    def is_rect(self):
        ''' We also store the upper triangular zeros. '''
        return True

    def snum_fe(self):
        return 1

    def snum_bs(self):
        return 1

    def check_numerical_stability(self):
        # check numerical stability of inverse
        ## conditional number
        cond = np.linalg.cond(self.dense_triag)
        msg = f'Conditional number of {self} is: {cond:.2e}'
        if cond > 1e8:
            eprint(msg)
        elif cond > 1e4:
            wprint(msg)
        else:
            print(msg)
        ## check solution error on 10 random inputs
        maxerr = 0
        for i in range(10):
            RANGE = 5
            exponent = np.random.random_sample(self.n)*(2*RANGE)-RANGE
            vec = np.exp(exponent)
            #vec = 1000*np.random.random_sample(self.n)-1
            sol = self.dense_inverse @ vec
            err = np.max( self.dense_triag @ sol - vec )
            if err > maxerr:
                maxerr = err
        if maxerr > 1e-5:
            wprint(f'\tInverse err on random input: = {maxerr:.2e}.')
        else:
            print(f'\tInverse err on random input: = {maxerr:.2e}.')

    def assigned_data(self):
        return sum(self.assigned_rows)

    def nnz(self):
        return np.count_nonzero(self.dense_triag) - self.n

    def color_dict(self, sq_dict, color):
        patches = []
        for ri in self.assigned_rows:
            for ci in range(ri):
                r = ri + self.rowa
                c = ci + self.cola
                if c in sq_dict[r]:
                    sq_dict[r][c].set_color(color)
                else:
                    box = plt.Rectangle((c-.5,r-.5), 1, 1, fc=color ,ec='lime',lw=1)
                    patches.append(box)
        return patches

    def schedule(self,cores=range(HARTS)):
        assert(len(self.assigned_rows) == 0)
        dist = [self.empty_copy() for h in cores]
        # compute list of rows for each worker
        tmp = list(cores) #H0, H1, H2
        tmpr = list(cores) #H0, H1, H2
        tmpr.reverse()
        tmp.extend(tmpr) #H0 H1 H2 H2 H1 H0
        for i,r in enumerate(range(self.rowz-self.rowa-1,0,-1)): # first col is empty
            core = tmp[i%(2*len(cores))]
            dprint(f'Assigning: r{r} H{core}\t',end='')
            dist[core].assigned_rows.append(r)
        dprint()
        return dist

    def codegen(self,s,h):
        assrow = f'{self}_h{h}_assigned_rows'
        mat = f'{self}_mat'
        argstruct = f'{self}_h{h}_args'

        lsolve = f'diaginv_lsolve(&{argstruct})'
        ltsolve = f'diaginv_ltsolve(&{argstruct})'
        #{self.n},{self.rowa},{mat},{assrow},{len(self.assigned_rows)}'
        dat = {}
        args = f'{self.n}, {self.rowa}, {mat}, {assrow}, {len(self.assigned_rows)}'
        dat[assrow] = np.array(self.assigned_rows,dtype=np.uint16)
        dat[mat] = self.dense_inverse
        dat[argstruct] = f'Diaginv {argstruct} = '+'{'+ args + '};\n'
        return (lsolve,ltsolve),dat

class Fold(DiagInv,Tile):
    REDUCES = True
    def empty_copy(self):
        raise NotImplementedError()
        tmp = Collist(self.rowa,self.rowz,self.cola,self.colz)
        return Fold(tmp)

    def schedule(self,cores=range(HARTS)):
        assert(len(self.assigned_rows) == 0)
        raise NotImplementedError()
        cols = []
        for row in range(1,self.n):
            val = self.Tinv[row][0:row]
            assert(len(val) == row)
            ri = [self.column_offset + i for i in range(row)]
            meta.append((row+self.column_offset,len(val),ri,val))
        meta = sorted(meta,key=lambda x: x[1],reverse=True)
        # schedule rows onto Hearts in parallel
        hearts = [i for i in range(HARTS)]
        # schedule greedely longest operations first to least busy heart
        slen = [0 for h in hearts]
        Pmeta = [[] for heart in hearts]
        PRi = [[] for heart in hearts]
        PRx = [[] for heart in hearts]
        if len(meta) == 0:
            return MetaRowSched(Pmeta,PRi,PRx,method=name,slen=slen)
        for j,(r,l,ri,rx) in enumerate(meta):
            h = min(hearts,key=slen.__getitem__)
            slen[h] += l
            Pmeta[h].append((r,l))
            PRx[h].extend(rx)
            PRi[h].extend(ri)
        name = self.__str__()
        return MetaRowSched(Pmeta,PRi,PRx,method=name,slen=slen)

    def to_Fold(self):
        ''' Convert triangular matrix to folded negative inverse'''
        T = -self.Tinv
        s = T.shape[0]
        (n,m) = (math.ceil(s/2),s-1)
        print(f"Folding {s}x{s} to {n}x{m}")
        F = np.zeros((n,m))
        for c in range(s-1):
            for r in range(c+1,s):
                if r == c:
                    pass # skip diagonal
                elif c < n:
                    F[c][r-1] = T[r][c]
                else:
                    F[r-n][c-n] = T[r][c]
        return F

class Csc:
    def __init__(self, Lp, Li, Lx, n, name=""):
        assert(type(name) is str)
        self.name = name
        assert(type(n) is int)
        self.n = n

        for i in [Lp,Li,Lx]:
            assert(type(i) is np.ndarray or type(i) is list)
        if type(Lp) is list:
            self.Lp = np.array(Lp,dtype=int)
        else:
            self.Lp = Lp
        if type(Li) is list:
            self.Li = np.array(Li,dtype=int)
        else:
            self.Li = Li
        if type(Lx) is list:
            self.Lx = np.array(Lx,dtype=float)
        else:
            self.Lx = Lx
        assert np.issubdtype(self.Lx.dtype, np.floating)
        assert np.issubdtype(self.Lp.dtype, np.integer)
        assert np.issubdtype(self.Li.dtype, np.integer)

        self.remove_zeros()

    def remove_zeros(self):
        #wprint("TESTING! DO NOT USE FOR PRODUCTION")
        #self.Lx[3] = 0
        numzeros = sum(self.Lx == 0)
        if numzeros > 0:
            print(f'Optimizing data: removing {numzeros} zeros in L matrix of {self.name}.')
            LpOpt = [0]
            LxOpt = []
            LiOpt = []
            for (a,b) in zip(self.Lp[:-1],self.Lp[1:]):
                for elem in range(a,b):
                    i = self.Li[elem]
                    x = self.Lx[elem]
                    if x != 0:
                        LxOpt.append(x)
                        LiOpt.append(i)
                LpOpt.append(len(LiOpt))
            self.Lp = np.array(LpOpt)
            self.Li = np.array(LiOpt)
            self.Lx = np.array(LxOpt)

    def n(self):
        return self.n

    def m(self):
        return len(self.Lp)-1

    def nnz(self):
        return len(self.Lx)

    def is_empty(self):
        return (self.nnz() == 0)

    def density(self):
        return self.nnz()/(self.n() * self.m())

    def __iter__(self):
        return Liter(self.Lp,self.Li,self.Lx)

    def __repr__(self):
        return self.__str__()
    #    a = f'{self.name} has {self.nnz} nonzeros and dimension {self.n}x{self.m}.\n'
    #    a += f'\tLp={self.Lp}\n'
    #    a += f'\tLi={self.Li}\n'
    #    a += f'\tLx={self.Lx}'
    #    return a

    def __str__(self):
        if self.is_empty():
            a = f'CSC "{self.name}" is empty.'
        else:
            a = f'CSC "{self.name}" has {self.nnz()} nonzeros and dimension {self.n}x{self.m()}. Density {self.density():.4f}'
        return a

    def plot(self, multicore_coloring=False, numcores=9, diag=False, uselx=True):
        Lp = self.Lp
        Li = self.Li
        Lx = self.Lx
        n = len(Lp) - 1

        def get_plot_rect(x,y,diag=False,color=None):
            if color is not None:
                c = color
            elif diag:
                c = (0.4,0.4,0.4)
            else:
                c = (0,0,0)
            return plt.Rectangle((x-.5+border,y-.5+border), 1+2*border, 1+2*border, fc=c,ec=c,lw=linewidth)

        # Color values
        vmin=np.min(np.abs(Lx))
        vmax=np.max(np.abs(Lx))
        if vmin > 0.0:
            norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        else: 
            wprint(f'{sum(np.abs(Lx)==0)} zeros in data: consider optimizing. Choosing linear color scale.')
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        #normalizer = np.vectorize(norm)
        #LxC = normalizer(Lx)
        cmap = plt.cm.rainbow

        # Dictionary for holding artists
        sq_dict = {row:{} for row in range(n)} # dictionary containing all squares
        border = 0
        linewidth = 5e-2
        for col in range(n):
            # get indices:
            first = Lp[col]
            last = Lp[col+1]
            #print(f'col={} {first}:{last}')
            for idx in range(first,last):
                row = Li[idx]
                val = np.abs(Lx[idx])
                color = cmap(norm(val)) if uselx else None
                sq_dict[row][col] = get_plot_rect(col,row,color=color)
        # Add Diagonals
        if diag:
            for i in range(n):
                sq_dict[i][i] = get_plot_rect(i,i,diag=True)


        fig, ax = plt.subplots(figsize=(7,7))
        print(f"Matrix dimension = {n}x{n}")
        ax.set(xlim=(-0.5, n-0.5), ylim=(-0.5, n-0.5))
        # Add line diagonal
        ax.plot([-0.5,n-0.5],[-0.5,n-0.5],color='k',linewidth=0.5)

        for row in sq_dict:
            for col in sq_dict[row]:
                ax.add_patch(sq_dict[row][col])

        if multicore_coloring:
            elems = []
            for i,c in enumerate(color_palette):
                patch = Patch(facecolor=c, edgecolor=None, label=f'Core {i}')
                elems.append(patch)
            ax.legend(handles=elems, loc='upper right')

        ax.set_ylim(ax.get_ylim()[::-1]) #invert axis
        ax.xaxis.tick_top()
        ax.yaxis.tick_left()
        ax.axes.set_aspect('equal')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if uselx:
            fig.colorbar(sm)
        su = 0
        for r in sq_dict:
            su += len(sq_dict[r])
        assert(su == self.nnz())
        return (fig,ax,sq_dict)

class Triag(Csc):

    def density(self):
        ''' Override density. Since it is better given its a lower triang matrix. '''
        underdiag = int((self.n)*(self.n-1)/2)
        return self.nnz()/underdiag

    def cut(self,cut,cut2):
        if DEBUG:
            print(f'Cutting columns {cut} to {cut2}.')
        assert(0 <= cut < self.n)
        assert(1 < cut2 <= self.n)
        assert(cut < cut2) # so at least we cut one column
        # cut is the first column we take
        # then we take all cols upuntil but excluding cat2

        # compute Lbot
        LbotP = [0]; LbotI = []; LbotX = []
        name = f'Lbot from row {cut} to {self.n}'
        for c in range(cut,self.n):
            for elem in range(self.Lp[c],self.Lp[c+1]):
                LbotI.append(self.Li[elem])
                LbotX.append(self.Lx[elem])
            LbotP.append(len(LbotX))
        Lbot = Triag(LbotP,LbotI,LbotX,self.n-cut,name)

        # compute rectangle
        Rp = [0]; Ri = []; Rx = []
        for c in range(0,cut):
            for elem in range(self.Lp[c],self.Lp[c+1]):
                r = self.Li[elem]
                if (cut <= r < cut2):
                    Ri.append(r)
                    Rx.append(self.Lx[elem])
            Rp.append(len(Rx))
        R = Rect(Rp,Ri,Rx,len(Rp)-1,cut,cut2-cut,name=f"Rectange from 0 to {cut}")

        # compute Ltop -> the remainder after the cut
        LtopP = [0]; LtopI = []; LtopX = []
        name = f'Ltop from row 0 to {cut}'
        for c in range(0,cut):
            for elem in range(self.Lp[c],self.Lp[c+1]):
                r = self.Li[elem]
                if(r < cut):
                    LtopI.append(r)
                    LtopX.append(self.Lx[elem])
            LtopP.append(len(LtopX))
        Ltop = Triag(LtopP,LtopI,LtopX,cut,name)
        return (Ltop,R,Lbot)



class Liter:
    ''' Iterator class for L matrix. Mainly used to block the matrix'''
    class cit:
        ''' column iterator for L matrix '''
        def __init__(self,col,i,x):
            self.col = col
            self.i = i
            self.x = x
            self.num = 0; #num of next element to return

        def has_next(self):
            return self.num < len(self.i)

        def consume(self):
            if not self.has_next():
                raise StopIteration
            val = self.lookahead()
            self.num+=1
            return val

        def lookahead(self):
            if self.has_next():
                return (self.i[self.num], self.x[self.num])
            else:
                return (None,None)

    def __init__(self,Lp,Li,Lx):
        self.Lx = Lx
        self.Li = Li
        self.Lp = Lp
        self.cc = 0 #current column

    def __iter__(self):
        return self

    def __next__(self):
        self.cc += 1
        if (len(self.Lp) > self.cc):
            start = self.Lp[self.cc-1]
            end = self.Lp[self.cc]
            i = self.Li[start:end]
            x = self.Lx[start:end]
            return self.cit(self.cc-1,i,x)
        else:
            raise StopIteration


def sparse2blocked(Lp,Li,Lx):
    ''' Put together 2 data elements in one block.
        If there are not two matrix elements direct below each other
        add fill in zeros.'''
    newLi = []
    newLx = []
    newLp = [0]

    lit = Liter(Lp,Li,Lx)
    for cit in lit: # go through col iterators
        while(cit.has_next()):
            (idx,val) = cit.consume()
            if(idx % 2 == 0):
                nidx,nval = cit.lookahead()
                if nidx is not None and nidx == idx+1:
                    newLi.append(idx//2)
                    newLx.append(val)
                    cit.consume()
                    newLx.append(nval)
                    continue
                else:
                    newLi.append(idx//2)
                    newLx.append(val)
                    newLx.append(0)
            else:
                newLi.append(idx//2)
                newLx.append(0)
                newLx.append(val)
        newLp.append(len(newLi))
    Li = np.array(newLi,dtype=Li.dtype)
    Lx = np.array(newLx,dtype=type(Lx[0]))
    Lp = np.array(newLp,dtype=Lp.dtype)
    return (Lp,Li,Lx)


def sparse2blockedTwice(Lp,Li,Lx):
    ''' Put together 4 data elements in one block.'''
    # The idea here is to call the blocking function twice to achieve
    #  blocking of 4 data elements together.
    newLp,newLi,newLx = sparse2blocked(Lp,Li,Lx)
    # put two adjacent data elements in one
    newLxBlocked = []
    l = range(len(newLx)//2)
    for a,b in zip([newLx[2*i] for i in l],[newLx[2*i+1] for i in l]):
        newLxBlocked.append((a,b))
    # block again
    newLp2,newLi2,newLx2 = sparse2blocked(newLp,newLi,newLxBlocked)
    ix = 0
    LxFp16 = np.empty((len(newLx2)*2),dtype=bfloat16)
    for i in newLx2:
        if type(i) is tuple:
            a,b = i
        elif type(i) is int:
            a = 0
            b = 0
        else:
            raise ValueError(f"Unexpected type result from blocking: {type(i)}.")
        LxFp16[ix] = a
        ix += 1
        LxFp16[ix] = b
        ix += 1
    return (newLp2,newLi2,LxFp16)
