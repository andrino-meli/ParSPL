import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
import math
from general import svdinvert, wprint, DEBUG, eprint, dprint, HARTS
from bfloat16 import bfloat16

# select if want to plot L**-1
NUM_CORES = 8

#color_palette = [(0xfc,0xda,0x8f), (0x28,0xc1,0xbc), (0x06,0x67,0xdd), (0x94,0xc1,0xe0), (0x88,0xd6,0x2f), (0xf4,0x70,0xbf), (0xdd,0x6c,0x58), (0x18,0x9b,0x42), (0x0e,0x8c,0x73)]
#color_palette = [(0x71,0x56,0x60), (0x84,0x64,0x70), (0x56,0x60,0x71), (0x72,0x7f,0x96), (0xc3,0x9f,0x72), (0xd5,0xbf,0x95), (0x66,0x77,0x62), (0x87,0x9e,0x82)]
color_palette = [
        (0x1e,0x4a,0x28),
        (0x51,0xa1,0x6a),
        (0x77,0xab,0x75),
        (0x9c,0xb5,0x7f),
        (0xe6,0xc9,0x94),
        (0xcc,0x7a,0x3d),
        (0xc8,0x67,0x39),
        (0xc4,0x53,0x35),
        ]
color_palette = [(r/255,g/255,b/255) for (r,g,b) in color_palette]
color_palette.reverse()


class Tile:
    ''' Abstract class to contain a tile. Inherit this by a tile class that is also a cointainer for data.'''
    def __init__(self,rowa,rowz,cola,colz):
        self.rowa = rowa # start row
        self.rowz = rowz # end row
        self.cola = cola # start col
        self.colz = colz # end col

    def empty_copy(self):
        # call super class __init__
        new = type(self)(self.rowa,self.rowz,self.cola,self.colz)
        return new

    def is_diag(self):
        return self.rowa==self.cola and self.rowz==self.colz

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
        return f'{self.classname()} {self.rowa}-{self.rowz}r {self.cola}-{self.colz}c'

    def show_on_plot(self,ax,number=None):
        plotobjs = []
        if self.is_diag():
            color = 'r'
            x = [self.cola-0.5,self.colz-0.5,self.cola-0.5,self.cola-0.5]
            y = [self.rowa-0.5,self.rowz-0.5,self.rowz-0.5,self.rowa-0.5]
        else:
            color = 'b'
            x = [self.cola-0.5,self.colz-0.5,self.colz-0.5,self.cola-0.5,self.cola-0.5]
            y = [self.rowa-0.5,self.rowa-0.5,self.rowz-0.5,self.rowz-0.5,self.rowa-0.5]
        lines = ax.plot(x,y,color=color,linewidth=1.5)
        plotobjs.extend(lines)
        if number is not None:
            xmid = -0.5 + self.cola + 0.5*(self.colz-self.cola) 
            ymid = -0.5 + self.rowa + 0.5*(self.rowz-self.rowa)
            label = f' s{number}\n{self.classname()}'
            txt = ax.text(xmid,ymid,label,fontsize=17,color=color,ha='center',va='center')
            plotobjs.append(txt)
        return plotobjs


class Collist(Tile,dict):
    # dictionary containting data:
    #  key is the column number
    #  value is a tuple of the two lists: ([Li],[Lx])
    #  empty columns are forbiden
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

    def empty(self):
        return len(self) == 0

    def merge(self,other):
        if not isinstance(other,Collist):
            raise NotImplementedError(f'Trying to merge {type(self)} with {type(other)}')
        # TODO: assert we are directly below each other:
        #       currently we assume the merging is done correctly according to a dep. tree.
        #if:
        #else:
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

class Empty(Tile):
    def __init__(self,collist):
        assert(isinstance(collist,Collist))
        self.rowa = collist.rowa
        self.rowz = collist.rowz
        self.cola = collist.cola
        self.colz = collist.colz

    def nnz(self):
        return 0

    def empty(self):
        return True

class Fold(Tile):
    def __init__(self,collist):
        assert(isinstance(collist,Collist))
        self.rowa = collist.rowa
        self.rowz = collist.rowz
        self.cola = collist.cola
        self.colz = collist.colz

        # get dimensions
        self.n = self.rowz-self.rowa
        self.offset = self.rowa
        assert(self.is_diag())
        assert(self.n > 1) # 1x1 lower triag is just empty

        # fill matrix with data
        self.dense_triag = np.eye(self.n)
        for col,(Li,Lx) in collist.items():
            for i,x in zip(Li,Lx):
                r,c = col-self.offset, i - self.offset
                assert(r < c)
                self.dense_triag[r][c] = x

        # invert
        self.dense_inverse = np.linalg.inv(self.dense_triag)

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
            vec = 1000*np.random.random_sample(self.n)-1
            sol = self.dense_inverse @ vec
            err = np.max( self.dense_triag @ sol - vec )
            if err > maxerr:
                maxerr = err
        if maxerr > 1e-5:
            wprint(f'\tInverse err on random input: = {maxerr:.2e}.')
        else:
            print(f'\tInverse err on random input: = {maxerr:.2e}.')

    def nnz(self):
        return np.count_nonzero(self.dense_triag) - self.n

    def empty_copy(self):
        tmp = Collist(self.rowa,self.rowz,self.cola,self.colz)
        return Fold(tmp)

    def color_dict(self, sq_dict, color):
        pass

    def metaSchedule(self):
        # Extract Rows and metadata without index data
        # TODO: generate different layout!
        meta = []
        for row in range(1,self.n):
            val = self.Tinv[row][0:row]
            assert(len(val) == row)
            ri = [self.column_offset + i for i in range(row)]
            meta.append((row+self.column_offset,len(val),ri,val))
        meta = sorted(meta,key=lambda x: x[1],reverse=True)
        # schedule rows onto Hearts in parallel
        hearts = [i for i in range(NUM_CORES)]
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


class MetaRowSched():
    ''' Meta Data based Row Scheduling of Data onto Hearts. By now I am like facebook: deprived of name ideas. '''
    def __init__(self,Pmeta,PRi,PRx,slen=None,method=''):
        self.Pmeta = Pmeta
        self.PRi = PRi
        self.PRx = PRx
        self.method = method
        if slen is None:
            raise NotImplementedError
        self.slen = slen

    def color_dict(self, sq_dict,fillin=False):
        ''' Color sq_dict according to schedule '''
        patchlist = []
        for h in HEARTS:
            idxoff = 0
            for r,l in self.Pmeta[h]:
                for idx in range(idxoff,idxoff+l):
                    c = self.PRi[h][idx]
                    # for dense matrix we suffer fill-in.
                    if fillin and c not in sq_dict[r]:
                        box = plt.Rectangle((c+1-.5,r+1-.5), 1, 1, fc=color_palette[h],ec='b',lw=1)
                        patchlist.append(box)
                    else:
                        sq_dict[r][c].set_color(color_palette[h])
                idxoff += l
        if fillin:
            return patchlist
    
    def isEmpty(self):
        return sum(self.slen) == 0

    def __str__(self):
        a = ''
        a += self.method+'\n'
        if self.isEmpty():
            return a+'  empty\n'
        for h in HEARTS:
            a += f"  H{h} sum{self.slen[h]} "
            for (r,l) in self.Pmeta[h]:
                a += f' r{r}: {l}  '
            a += '\n'
        return a

    def strH(self,h):
        if self.isEmpty():
            return 'empty'
        a = f"sum{self.slen[h]}  "
        for (r,l) in self.Pmeta[h]:
            a += f'  r{r}: {l}'
        return a
