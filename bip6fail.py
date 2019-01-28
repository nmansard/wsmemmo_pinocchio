from robots import loadTalosLegs
from scipy.optimize import fmin_slsqp
import pinocchio
from pinocchio.utils import *
from numpy.linalg import norm,inv,pinv,eig,svd

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T

robot   = loadTalosLegs()
robot.initDisplay(loadModel=True)

class OptimProblem:
    def __init__(self,rmodel,rdata,gview=None):
        self.rmodel = rmodel
        self.rdata = rdata

        self.refL = pinocchio.SE3(eye(3), np.matrix([ 0., 1.5, 1.]).T )
        self.idL = rmodel.getFrameId('left_sole_link')  # ID of the robot object to control

        self.refR = pinocchio.SE3(eye(3), np.matrix([ 0., -1.5, 0.]).T )
        self.idR = rmodel.getFrameId('right_sole_link')# ID of the robot object to control

        self.initDisplay(gview)

    def cost(self,x):
        q = a2m(x)
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)

        refMl = self.refL.inverse()*self.rdata.oMf[self.idL]
        residualL = m2a(pinocchio.log(refMl).vector)
        refMr = self.refR.inverse()*self.rdata.oMf[self.idR]
        residualR = m2a(pinocchio.log(refMr).vector)

        self.residuals = np.concatenate([residualL,residualR])
        return sum( self.residuals**2 )
    
    # --- BLABLA -------------------------------------------------------------
    def initDisplay(self,gview):
        if gview is None: return 
        self.gview = gview
        self.gobjR = "world/targetR"
        self.gobjL = "world/targetL"
        self.gview.addBox(self.gobjR,.1,.03,.03,[1,0,0,1])
        self.gview.addBox(self.gobjL,.1,.03,.03,[0,1,0,1])
        self.gview.applyConfiguration(self.gobjR,se3ToXYZQUAT(self.refR))
        self.gview.applyConfiguration(self.gobjL,se3ToXYZQUAT(self.refL))
        self.gview.refresh()
    def callback(self,x):
        import time
        q = a2m(x)
        robot.display(q)
        time.sleep(1e-2)
     
pbm = OptimProblem(robot.model,robot.data,robot.viewer.gui)

x0  = m2a(robot.q0)
result = fmin_slsqp(x0       = x0,
                    func     = pbm.cost,
                    callback = pbm.callback)
qopt = a2m(result)
