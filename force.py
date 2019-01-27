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
        self.jidL = rmodel.frames[self.idL].parent
        self.jidR = rmodel.frames[self.idR].parent
        self.initDisplay(gview)
        self.refQ = rmodel.neutralConfiguration

        self.x = None
        self.xdiff = None
        self.ncost = self.rmodel.nv-6
        self.residuals = np.zeros(self.ncost)

        self.neq = 12+rmodel.nv
        self.eq = np.zeros(self.neq)
        self.Jeq = np.zeros([self.neq, self.rmodel.nv])

        # configurations are represented as velocity integrated from this point.
        self.q0 = rmodel.neutralConfiguration
                
    def vq2q(self,vq):   return pinocchio.integrate(rmodel,self.q0,vq)
    def q2vq(self,q):    return pinocchio.difference(rmodel,self.q0,q)

    def x2qf(self,x):
        vq = a2m(x[:self.rmodel.nv])
        q =  self.vq2q(vq)
        tauq = a2m(x[self.rmodel.nv:2*self.rmodel.nv])
        fr = a2m(x[2*self.rmodel.nv:2*self.rmodel.nv+6])
        fl = a2m(x[2*self.rmodel.nv+6:2*self.rmodel.nv+12])
        return q,tauq,fr,fl
        
    def costQ(self,x):
        q,tauq,fr,fl = self.x2qf(x)
        self.residuals[:] = pinocchio.difference(self.rmodel,self.refQ,q)[6:].flat
        return sum( self.residuals**2 )

    def constraint_leftfoot(self,x,nc=0):
        q,tauq,fr,fl = self.x2qf(x)
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMl = self.refL.inverse()*self.rdata.oMf[self.idL]
        self.eq[nc:nc+6] = m2a(pinocchio.log(refMl).vector)
        return self.eq[nc:nc+6].tolist()

    def constraint_rightfoot(self,x,nc=0):
        q,tauq,fr,fl = self.x2qf(x)
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        refMr = self.refR.inverse()*self.rdata.oMf[self.idR]
        self.eq[nc:nc+6] = m2a(pinocchio.log(refMr).vector)
        return self.eq[nc:nc+6].tolist()

    def constraint_dyn(self,x,nc=0):
        '''
        M aq + b(q,vq) + g(q) = tau_q + J(q)^T f
        = rnea(q,vq=0,aq=0,fs)
        '''
        q,tauq,fr,fl = self.x2qf(x)

        # Forces should be stored in RNEA-compatible structure
        forces = pinocchio.StdVect_Force()
        for i in range(self.rmodel.njoints): forces.append(pinocchio.Force.Zero())
        # Forces should be expressed at the joint for RNEA, while we store them
        # at the frame in the optim problem. Convert.
        jMr = rmodel.frames[self.idR].placement
        forces[self.jidR] = jMr*pinocchio.Force(fr)
        jMl = rmodel.frames[self.idL].placement
        forces[self.jidL] = jMl*pinocchio.Force(fl)

        #q = self.rmodel.neutralConfiguration
        aq = vq = zero(self.rmodel.nv)
        tauref = pinocchio.rnea(self.rmodel,self.rdata,q,vq,aq,forces)
        self.eq[nc:nc+self.rmodel.nv] = (tauref-tauq).flat
        return self.eq[nc:nc+self.rmodel.nv].tolist()
        
    def constraint(self,x):
        self.constraint_rightfoot(x,0)
        self.constraint_leftfoot(x,6)
        self.constraint_dyn(x,12)
        return self.eq.tolist()
        
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




    
robot.q0 = rmodel.neutralConfiguration

pbm = OptimProblem(rmodel,rdata,gview)

q = pinocchio.randomConfiguration(rmodel)
vq = pbm.q2vq(q)
fr = rand(6)
fl = rand(6)
tauq = rand(rmodel.nv)


#q = rmodel.neutralConfiguration.copy()
vq = pbm.q2vq(q)
fr = zero(6)
fl = zero(6)
tauq = zero(rmodel.nv)

                                    
x0 = m2a(np.concatenate([vq,tauq,fr,fl]))

#res = fmin_slsqp(func=pbm.costQ,x0=x0,f_eqcons=pbm.constraint,epsilon=1e-5,callback=pbm.callback)
res = fmin_slsqp(x0=x0,
                 func=pbm.costQ,
                 #func=lambda x: pbm.costQ(x) + sum([c**2 for c in pbm.constraint(x)]),x0=x0,
                 f_eqcons=pbm.constraint,
                 epsilon=1e-7,callback=pbm.callback,iter=1000)

q,tau,fr,fl = pbm.x2qf(res)
rnea = pinocchio.rnea(rmodel,rdata,q,zero(18),zero(18))
assert( norm( tau-rnea) <1e-5 )
stophere



def cost(tau):
    tau = a2m(tau)
    q = rmodel.neutralConfiguration
    v = zero(rmodel.nv)
    a = zero(rmodel.nv)
    res = m2a(pinocchio.rnea(rmodel,rdata,q,v,a)-tau)
    return sum( res**2 )
def constraint(tau):
    tau = a2m(tau)
    q = rmodel.neutralConfiguration
    v = zero(rmodel.nv)
    a = zero(rmodel.nv)
    res = m2a(pinocchio.rnea(rmodel,rdata,q,v,a)-tau)
    return res.tolist()

res = fmin_slsqp(func=lambda x: sum(x**2),x0=m2a(rand(18)),
                 f_eqcons = constraint,
                 epsilon=1e-5)


