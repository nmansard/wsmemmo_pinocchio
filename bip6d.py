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
    def __init__(self,rmodel,rdata,gview=None):
        self.rmodel = rmodel
        self.rdata = rdata
        self.refL = pinocchio.SE3(eye(3), np.matrix([ 0., .3, 0.]).T )
        self.refR = pinocchio.SE3(eye(3), np.matrix([ 0., -.3, 0.]).T )
        self.idL = rmodel.getFrameId('left_sole_link')  # ID of the robot object to control
        self.idR = rmodel.getFrameId('right_sole_link')# ID of the robot object to control
        self.initDisplay(gview)
        self.refQ = rmodel.neutralConfiguration

        self.x = None
        self.xdiff = None
        self.ncost = self.rmodel.nv-6
        self.residuals = np.zeros(self.ncost)

        self.neq = 12
        self.eq = np.zeros(self.neq)
        self.Jeq = np.zeros([self.neq, self.rmodel.nv])

        # configurations are represented as velocity integrated from this point.
        self.q0 = rmodel.neutralConfiguration

    def vq2q(self,vq):   return pinocchio.integrate(rmodel,self.q0,vq)
    def q2vq(self,q):    return pinocchio.difference(rmodel,self.q0,q)
    def dq_dv(self,vq):  return pinocchio.dIntegrate(rmodel,self.q0,vq)[1]
        
    def pinocchioCalc(self,x):
        if x is not self.x:
            self.x = x
            vq = a2m(x)
            q = self.vq2q(vq)
            pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
            pinocchio.updateFramePlacements(self.rmodel,self.rdata)

    def pinocchioCalcDiff(self,x):
        if x is not self.xdiff:
            self.xdiff = x
            vq = a2m(x)
            q = self.vq2q(vq)
            pinocchio.computeJointJacobians(self.rmodel,self.rdata,q)
            pinocchio.updateFramePlacements(self.rmodel,self.rdata)
            self.Q_v = self.dq_dv(vq)
        
    def costQ(self,x):
        self.pinocchioCalc(x)
        q = self.vq2q(a2m(x))
        self.residuals[:] = pinocchio.difference(self.rmodel,self.refQ,q)[6:].flat
        return sum( self.residuals**2 )

    def dCostQ_dx(self,x):
        self.pinocchioCalcDiff(x)
        q = self.vq2q(a2m(x))
        # ddiff_dx2(x1,x2) = dint_dv(x1,x2-x1)^-1
        # ddiff_dq( refQ,q) = dint_dv(refQ,q-refQ)
        dq = pinocchio.difference(self.rmodel,self.refQ,q)
        dDiff = inv(pinocchio.dIntegrate(rmodel,self.refQ,dq)[1])
        grad = dDiff[6:,:].T*dq[6:]
        return m2a(grad)
        
    def constraint_leftfoot(self,x,nc=0):
        self.pinocchioCalc(x)
        refMl = self.refL.inverse()*self.rdata.oMf[self.idL]
        self.eq[nc:nc+6] = m2a(pinocchio.log(refMl).vector)
        return self.eq[nc:nc+6].tolist()

    def constraint_rightfoot(self,x,nc=0):
        self.pinocchioCalc(x)
        refMr = self.refL.inverse()*self.rdata.oMf[self.idR]
        self.eq[nc:nc+6] = m2a(pinocchio.log(refMr).vector)
        return self.eq[nc:nc+6].tolist()

    def constraint(self,x):
        self.constraint_rightfoot(x,0)
        self.constraint_leftfoot(x,6)
        return self.eq.tolist()
    
    def dConstraint_dx_leftfoot(self,x,nc=0):
        self.pinocchioCalcDiff(x)
        rMl = self.refL.inverse()*self.rdata.oMf[self.idL]
        log_M = pinocchio.Jlog6(rMl)
        M_q = pinocchio.getFrameJacobian(self.rmodel,self.rdata,self.idL,LOCAL)
        self.Jeq[nc:nc+6,:] = log_M*M_q*self.Q_v
        return self.Jeq[nc:nc+6,:]

    def dConstraint_dx_rightfoot(self,x,nc=0):
        self.pinocchioCalcDiff(x)
        refMr = self.refL.inverse()*self.rdata.oMf[self.idR]
        log_M = pinocchio.Jlog6(refMr)
        M_q = pinocchio.getFrameJacobian(self.rmodel,self.rdata,self.idR,LOCAL)
        self.Jeq[nc:nc+6,:] = log_M*M_q*self.Q_v
        return self.Jeq[nc:nc+6,:]

    def dConstraint_dx(self,x):
        self.dConstraint_dx_rightfoot(x,0)
        self.dConstraint_dx_leftfoot(x,6)
        return self.Jeq
        
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
        print self.costQ(x),sum([ c**2 for c in self.constraint(x) ])




    
#for i,f in enumerate(rmodel.frames): print i,f.name
robot.q0 = pinocchio.randomConfiguration(rmodel)
robot.q0 = rmodel.neutralConfiguration

pbm = OptimProblem(rmodel,rdata,gview)
x0  = m2a(robot.q0)

# ------------------------------------------------------
def checkNumDiff(f,x,h=1e-6):
    f0 = np.array(f(x))
    nf,nx = len(f0),len(x)
    dx = np.zeros(nx)
    J = np.zeros([ nf,nx ])
    for i in range(nx):
        dx[i] = h
        J[:,i] = (np.array(f(x+dx))-f0)/h
        dx[i] = 0
    return J

x0  = m2a(pbm.q2vq(robot.q0))
x0  = np.random.rand(rmodel.nv)
Ja = pbm.dConstraint_dx(x0)
Jn = checkNumDiff(pbm.constraint,x0,h=1e-9)
assert(norm(Ja-Jn)<1e-4)
# ------------------------------------------------------


res = fmin_slsqp(func=pbm.costQ,x0=x0,f_eqcons=pbm.constraint,epsilon=1e-6,callback=pbm.callback)
res = fmin_slsqp(func=pbm.costQ,x0=x0,f_eqcons=pbm.constraint,fprime_eqcons=pbm.dConstraint_dx,callback=pbm.callback,iter=3)
qopt = pbm.vq2q(a2m(res))
