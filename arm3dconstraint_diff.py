from robots import loadTalosArm
from scipy.optimize import fmin_slsqp
import pinocchio
from pinocchio.utils import *
from numpy.linalg import norm,inv,pinv,eig,svd

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
LOCAL = pinocchio.ReferenceFrame.LOCAL
WORLD = pinocchio.ReferenceFrame.WORLD

robot   = loadTalosArm()
robot.initDisplay(loadModel=True)

class OptimProblem:
    def __init__(self,rmodel,gview=None):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()

        self.refEff = pinocchio.SE3( rotate('y',np.pi/4),                 # Target orientation
                                     np.matrix([ -.3, 0.5, 0.2 ]).T)     # Target position
        self.idEff = rmodel.getFrameId('gripper_left_fingertip_2_link')
        self.refQ = rmodel.neutralConfiguration

        self.initDisplay(gview)
        
    def cost(self,x):
        q = a2m(x)
        self.residuals = m2a(q-self.refQ)
        return .5*sum(self.residuals**2)

    def dCost_dx(self,x):
        q = a2m(x)
        dq = m2a(q-self.refQ)
        return dq
    
    def constraint(self,x):
        q = a2m(x)
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMeff = self.refEff.inverse()*self.rdata.oMf[self.idEff]
        self.eq = m2a(pinocchio.log(refMeff).vector)
        return self.eq.tolist()

    def dConstraint_dx(self,x):
        q = a2m(x)
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.computeJointJacobians(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMeff = self.refEff.inverse()*self.rdata.oMf[self.idEff]
        log_M = pinocchio.Jlog6(refMeff)
        M_q = pinocchio.getFrameJacobian(self.rmodel,self.rdata,self.idEff,LOCAL)
        self.Jeq = log_M*M_q
        return self.Jeq
        
    @property
    def bounds(self):
        # return [ (10*l,u) for l,u in zip(self.rmodel.lowerPositionLimit.flat,
        #                               self.rmodel.upperPositionLimit.flat) ]
        return [ (-10.,10) for i in range(self.rmodel.nq) ]
        
    def initDisplay(self,gview=None):
        self.gview = gview
        if gview is None: return
        self.gobj = "world/target6d"
        self.gview.addBox(self.gobj,.1,0.05,0.025,[1,0,0,1])
        self.gview.applyConfiguration(self.gobj,se3ToXYZQUAT(self.refEff))
        self.gview.refresh()

    def callback(self,x):
        import time
        q = a2m(x)
        robot.display(q)
        time.sleep(1e-2)


robot.q0 = robot.model.neutralConfiguration
pbm = OptimProblem(robot.model,robot.viewer.gui)

# --- NUMDIFF CHECK ------------------------------------
def numdiff(f,x,h=1e-6):
    f0 = f(x)
    nx,nf = len(x),len(f0)
    dx = np.zeros(nx)
    df_dx = np.zeros([nf,nx])
    for i in range(nx):
        dx[i] = h
        df_dx[:,i] = (f(x+dx)-f0)/h
        dx[i] = 0
    return df_dx

x = np.random.rand(robot.model.nq)*2-1

def costResiduals(x):
    pbm.cost(x)
    return pbm.residuals

assert( norm( pbm.dCost_dx(x) - np.dot( numdiff(costResiduals,x).T,costResiduals(x) ) ) <1e-6 )
assert( norm( pbm.dConstraint_dx(x) - numdiff(lambda x:np.array(pbm.constraint(x)),x) ) <1e-6 )

# --- NUMDIFF CHECK ------------------------------------

#x0  = np.random.rand(robot.model.nq)
x0 = np.array([ .7,.9,.8,.5,.9,.7,.1])

result = fmin_slsqp(x0       = x0,
                    func     = pbm.cost,
                    fprime   = pbm.dCost_dx,
                    f_eqcons = pbm.constraint,
                    fprime_eqcons = pbm.dConstraint_dx,
                    bounds   = pbm.bounds,
                    callback = pbm.callback)
qopt = a2m(result)
