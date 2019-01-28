from robots import loadTalosLegs
from scipy.optimize import fmin_slsqp
import pinocchio
from pinocchio.utils import *
from numpy.linalg import norm,inv,pinv,eig,svd

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
LOCAL = pinocchio.ReferenceFrame.LOCAL
WORLD = pinocchio.ReferenceFrame.WORLD

robot   = loadTalosLegs()

robot.initDisplay(loadModel=True)

class OptimProblem:
    def __init__(self,rmodel,gview=None):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.refL = pinocchio.SE3(eye(3), np.matrix([ 0., .3, 0.]).T )
        self.refR = pinocchio.SE3(eye(3), np.matrix([ 0., -.3, 0.]).T )
        self.idL = rmodel.getFrameId('left_sole_link')  # ID of the robot object to control
        self.idR = rmodel.getFrameId('right_sole_link')# ID of the robot object to control
        self.initDisplay(gview)
        self.refQ = rmodel.neutralConfiguration

        self.neq = 12
        self.eq = np.zeros(self.neq)
        self.Jeq = np.zeros([self.neq, self.rmodel.nv])

        # configurations are represented as velocity integrated from this point.
        self.q0 = rmodel.neutralConfiguration

        self.initDisplay(gview)
        
    def vq2q(self,vq):   return pinocchio.integrate(self.rmodel,self.q0,vq)
    def q2vq(self,q):    return pinocchio.difference(self.rmodel,self.q0,q)
    def dq_dv(self,vq):  return pinocchio.dIntegrate(self.rmodel,self.q0,vq)[1]
                
    def costQ(self,x):
        q = self.vq2q(a2m(x))
        self.residuals = m2a(pinocchio.difference(self.rmodel,self.refQ,q)[6:])
        return sum( self.residuals**2 )

    def dCostQ_dx(self,x):
        '''
        ddiff_dx2(x1,x2) = dint_dv(x1,x2-x1)^-1
        ddiff_dq( refQ,q) = dint_dv(refQ,q-refQ)
        '''
        q = self.vq2q(a2m(x))
        dq = pinocchio.difference(self.rmodel,self.refQ,q)
        dDiff = inv(pinocchio.dIntegrate(self.rmodel,self.refQ,dq)[1])
        grad = dDiff[6:,:].T*dq[6:]
        return m2a(grad)
        
    def constraint_leftfoot(self,x,nc=0):
        q = self.vq2q(a2m(x))
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMl = self.refL.inverse()*self.rdata.oMf[self.idL]
        self.eq[nc:nc+6] = m2a(pinocchio.log(refMl).vector)
        return self.eq[nc:nc+6].tolist()

    def constraint_rightfoot(self,x,nc=0):
        q = self.vq2q(a2m(x))
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMr = self.refR.inverse()*self.rdata.oMf[self.idR]
        self.eq[nc:nc+6] = m2a(pinocchio.log(refMr).vector)
        return self.eq[nc:nc+6].tolist()

    def constraint(self,x):
        self.constraint_rightfoot(x,0)
        self.constraint_leftfoot(x,6)
        return self.eq.tolist()
    
    def dConstraint_dx_leftfoot(self,x,nc=0):
        q = self.vq2q(a2m(x))
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        pinocchio.computeJointJacobians(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMl = self.refL.inverse()*self.rdata.oMf[self.idL]
        log_M = pinocchio.Jlog6(refMl)
        M_q = pinocchio.getFrameJacobian(self.rmodel,self.rdata,self.idL,LOCAL)
        Q_v = self.dq_dv(a2m(x))
        self.Jeq[nc:nc+6,:] = log_M*M_q*Q_v
        return self.Jeq[nc:nc+6,:]

    def dConstraint_dx_rightfoot(self,x,nc=0):
        q = self.vq2q(a2m(x))
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        pinocchio.computeJointJacobians(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMr = self.refR.inverse()*self.rdata.oMf[self.idR]
        log_M = pinocchio.Jlog6(refMr)
        M_q = pinocchio.getFrameJacobian(self.rmodel,self.rdata,self.idR,LOCAL)
        Q_v = self.dq_dv(a2m(x))
        self.Jeq[nc:nc+6,:] = log_M*M_q*Q_v
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
        import time
        q = self.vq2q(a2m(x))
        robot.display(q)
        time.sleep(1e-2)

pbm = OptimProblem(robot.model,robot.viewer.gui)

#x0  = np.random.rand(robot.model.nv)
x0 = np.array([0.6,0.5,0.6,0.3,0.0,0.8,0.7,0.5,0.8,0.9,0.0,0.3,0.8,0.6,0.2,0.7,0.4,0.1])

res = fmin_slsqp(x0=x0,
                 func=pbm.costQ,
                 fprime=pbm.dCostQ_dx,
                 f_eqcons=pbm.constraint,
                 fprime_eqcons=pbm.dConstraint_dx,
                 callback=pbm.callback,
                 iter=300)

qopt = pbm.vq2q(a2m(res))
