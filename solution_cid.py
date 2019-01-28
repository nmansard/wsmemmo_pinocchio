from robots import loadTalosLegs
from scipy.optimize import fmin_slsqp
import pinocchio
from pinocchio.utils import *
from numpy.linalg import norm,inv,pinv,eig,svd

robot   = loadTalosLegs()
rmodel  = robot.model
rdata   = robot.data

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
LOCAL = pinocchio.ReferenceFrame.LOCAL
WORLD = pinocchio.ReferenceFrame.WORLD

#robot.initDisplay(loadModel=True)
#gview = robot.viewer.gui
gview = None

class OptimProblem:
    def __init__(self,rmodel,rdata,q,vq,gview=None):
        self.rmodel = rmodel
        self.rdata = rdata

        self.idL = rmodel.getFrameId('left_sole_link')  # ID of the robot object to control
        self.idR = rmodel.getFrameId('right_sole_link')# ID of the robot object to control
        self.jidL = rmodel.frames[self.idL].parent
        self.jidR = rmodel.frames[self.idR].parent

        self.q = q.copy()
        self.vq = vq.copy()
        
    def x2vars(self,x):
        idx = 0
        nvar = self.rmodel.nv; tauq = a2m(x[idx:idx+nvar]); idx+=nvar
        nvar = 6             ; phir = a2m(x[idx:idx+nvar]); idx+=nvar
        nvar = 6             ; phil = a2m(x[idx:idx+nvar]); idx+=nvar
        return tauq,phir,phil
        
    def cost(self,x):
        tauq,fr,fl = self.x2vars(x)
        pinocchio.computeAllTerms(self.rmodel,self.rdata,self.q,self.vq)
        b = self.rdata.nle
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        Jr = pinocchio.getFrameJacobian(self.rmodel,self.rdata,self.idR,LOCAL)
        Jl = pinocchio.getFrameJacobian(self.rmodel,self.rdata,self.idL,LOCAL)
        self.residuals = m2a(b - tauq - Jr.T*fr - Jl.T*fl)
        return sum( self.residuals**2 )
        
    # --- BLABLA -------------------------------------------------------------
    def callback(self,x):
        pass
        
   
pbm = OptimProblem(rmodel,rdata,q=robot.q0,vq=zero(rmodel.nv))
x0 = np.zeros(robot.model.nv+12)

res = fmin_slsqp(x0=x0,
                 func=pbm.cost,
                 epsilon=1e-7,callback=pbm.callback,iter=1000)



