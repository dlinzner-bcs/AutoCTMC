from ctmc import ctmc

class ctbn(ctmc):

    def __init__(self, n, sdims,g, **kwargs):
        ctmc. __init__(**kwargs)
        self.n=n
        self.sdmis=sdims
        self.g = g

    def inv_amalgamate(self):
        for i in range(0,self.n):
            for x in range(0,self.sdims[i]):
                for y in range(0,self.sdims[i]):
                    q[x,y]=



