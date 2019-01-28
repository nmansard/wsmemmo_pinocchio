from robots import loadTalosArm
from scipy.optimize import fmin_slsqp
import pinocchio
from pinocchio.utils import *
from numpy.linalg import norm,inv,pinv,eig,svd

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T

robot   = loadTalosArm()
robot.initDisplay(loadModel=True)

class OptimProblem:
    def __init__(self,rmodel,rdata,gview=None):
        self.rmodel = rmodel
        self.rdata = rdata
        self.refEff = [ .3, 0.3, 0.3 ]     # Target position
        self.idEff = rmodel.getFrameId('gripper_left_fingertip_2_link')
        self.refQ = rmodel.neutralConfiguration
        self.initDisplay(gview)
        
    def cost(self,x):
        q = a2m(x)
        self.residuals = m2a(q-self.refQ)
        return sum(self.residuals**2)

    def constraint(self,x):
        q = a2m(x)
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        M = self.rdata.oMf[self.idEff]
        self.eq = m2a(M.translation) - self.refEff
        return self.eq.flat

    @property
    def bounds(self):
        return [ (10*l,u) for l,u in zip(self.rmodel.lowerPositionLimit.flat,
                                      self.rmodel.upperPositionLimit.flat) ]

    def initDisplay(self,gview=None):
        self.gview = gview
        if gview is None: return
        self.gobj = "world/target3d"
        self.gview.addSphere(self.gobj,.03,[1,0,0,1])
        self.gview.applyConfiguration(self.gobj,self.refEff+[0,0,0,1])
        self.gview.refresh()

    def callback(self,x):
        import time
        q = a2m(x)
        robot.display(q)
        time.sleep(1e-2)

pbm = OptimProblem(robot.model,robot.model.createData(),robot.viewer.gui)

x0  = m2a(robot.q0)
result = fmin_slsqp(x0       = x0,
                    func     = pbm.cost,
                    f_eqcons = pbm.constraint,
                    callback = pbm.callback)
qopt = a2m(result)
