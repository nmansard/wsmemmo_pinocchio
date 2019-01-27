from robots import loadTalosArm
from scipy.optimize import fmin_slsqp
import pinocchio
from pinocchio.utils import *
from numpy.linalg import norm,inv,pinv,eig,svd

robot   = loadTalosArm()
rmodel  = robot.model
rdata   = robot.data
robot.q0= rand(rmodel.nq)

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T

robot.initDisplay(loadModel=True)
gview = robot.viewer.gui

class OptimProblem:
    def __init__(self,rmodel,rdata,gview=None):
        self.rmodel = rmodel
        self.rdata = rdata
        self.ref = [ .3, 0.3, 0.3 ]    # Target position
        self.effId = -1                 # ID of the robot object to control
        self.gview = gview
        if self.gview:
            self.gobj = "world/target3d"
            self.gview.addSphere(self.gobj,.03,[1,0,0,1])
            self.gview.applyConfiguration(self.gobj,self.ref+[0,0,0,1])
            self.gview.refresh()
    def cost3(self,x):
        q = a2m(x)
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        M = rdata.oMi[self.effId]
        cost = norm(m2a(M.translation) - self.ref)
        print cost
        return cost


pbm = OptimProblem(rmodel,rdata,gview)
x0  = m2a(robot.q0)

res = fmin_slsqp(func=pbm.cost3,x0=x0)
qopt = a2m(res)
