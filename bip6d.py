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

robot.initDisplay(loadModel=True)
gview = robot.viewer.gui

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
        
    def pinocchioCalc(self,x):
        if x is not self.x:
            self.x = x
            q = a2m(x)
            pinocchio.forwardKinematics(self.rmodel,self.rdata,q)
            pinocchio.updateFramePlacements(self.rmodel,self.rdata)

    def pinocchioCalcDiff(self,x):
        if x is not self.xdiff:
            self.xdiff = x
            q = a2m(x)
            pinocchio.computeJointJacobians(self.rmodel,self.rdata)
            pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        
    def costQ(self,x):
        self.pinocchioCalc(x)
        q = a2m(x)
        self.residuals[:] = pinocchio.difference(self.rmodel,q,self.refQ)[6:].flat
        return sum( self.residuals**2 )
        
    def constraint_leftfoot(self,x):
        self.pinocchioCalc(x)
        q = a2m(x)
        Ml = self.rdata.oMf[self.idL]
        self.eq[:6] = m2a(pinocchio.log(Ml.inverse()*self.refL).vector)
        return self.eq[:6].tolist()

    def constraint_rightfoot(self,x):
        self.pinocchioCalc(x)
        q = a2m(x)
        Mr = self.rdata.oMf[self.idR]
        self.eq[6:12] = m2a(pinocchio.log(Mr.inverse()*self.refR).vector)
        return self.eq[6:12].tolist()

    def constraint(self,x):
        self.constraint_rightfoot(x)
        self.constraint_leftfoot(x)
        return self.eq.tolist()
    
    def Jeq_leftfoot(self,x):
        self.pinocchioCalcDiff(x)
        Rl = self.rdata.oMf[self.idL].rotation
        J6 = pinocchio.getFrameJacobian(rmodel,rdata,self.idL,pinocchio.ReferenceFrame.LOCAL)
        self.Jeq[:3,:] = Rl*J6[:3,:]
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
        pass            


def checkNumDiff(f,J,x):
    pass

    
#for i,f in enumerate(rmodel.frames): print i,f.name
robot.q0 = pinocchio.randomConfiguration(rmodel)
robot.q0 = rmodel.neutralConfiguration

pbm = OptimProblem(rmodel,rdata,gview)
x0  = m2a(robot.q0)

#res = fmin_slsqp(func=pbm.costQ,x0=x0,f_eqcons=pbm.constraint,epsilon=1e-6)
#qopt = a2m(res)

''' 
H =
 w  -z   y
 z   w  -x
-y   x   w
-x  -y  -z


[i]w = H*delta_q  

''' 

q = rmodel.neutralConfiguration.copy()
q.flat[3:7] = [.1,.2,.3,.4]
q[3:7] /= norm(q[3:7])
dq = q*0
h = 1e-6
pinocchio.forwardKinematics(rmodel,rdata,q)
M0 = rdata.oMi[1].copy()
J= zero([6,4])
for i in range(3,7):
    dq[i] = h
    pinocchio.forwardKinematics(rmodel,rdata,q+dq)
    dq[i]=0
    J[:,i-3] = pinocchio.log(M0.inverse()*rdata.oMi[1]).vector/h


def ch(q4):
    q = rmodel.neutralConfiguration.copy()
    q.flat[3:7] = q4
    q[3:7] /= norm(q[3:7])
    print q[3:7].T
    dq = q*0
    h = 1e-6
    pinocchio.forwardKinematics(rmodel,rdata,q)
    M0 = rdata.oMi[1].copy()
    J = zero([6,4])
    for i in range(3,7):
        dq[i] = h
        pinocchio.forwardKinematics(rmodel,rdata,q+dq)
        dq[i]=0
        J[:,i-3] = pinocchio.log(M0.inverse()*rdata.oMi[1]).vector/h
    return J


    

pinocchio.computeJointJacobians(rmodel,rdata,q)
J3=pinocchio.getJointJacobian(rmodel,rdata,1,LOCAL)[3:,3:6]

def w_q(q):
    if isinstance(q,np.ndarray): q = v2q(q)
    return pinocchio.log3(q.matrix())

v2q = lambda v_: pinocchio.Quaternion(v_[3,0],*v_[1:].flat)
q2v = lambda q_: q_.coeffs()

def dw_dq(q,h=1e-6):
    dq = zero(4)
    J = zero([3,4])
    for i in range(4):
        dq[i]=h
        J[:,i] = pinocchio.log3( v2q(q).matrix().T*v2q(q+dq).matrix() )/h
        dq[i]=0
    return J
'''
dw = dw/dq dq
R(q+dq) = R(q) + exp(dw)
'''
# q = rand(4); q/=norm(q)
# dq = rand(4)*1e-3
# assert( norm( pinocchio.log3(inv(v2q(q+dq).matrix()) \
#                              * v2q(q).matrix()*pinocchio.exp3(dw_dq(q)*dq)) )<1e-1 )


# q = pinocchio.randomConfiguration(rmodel)
# w = rand(3)*1e-2
# vq = zero(rmodel.nv); vq[3:6] = w

# pinocchio.forwardKinematics(rmodel,rdata,q)
# M = rdata.oMi[1].copy()
# pinocchio.forwardKinematics(rmodel,rdata,pinocchio.integrate(rmodel,q,vq))
# Md = rdata.oMi[1].copy()
# assert( norm( M.rotation*pinocchio.exp3(w) - Md.rotation ) <1e-6 )

# dq = zero(rmodel.nq)
# dq4 = rand(4) * 1e-4
# dq[3:7] = dq4

# pinocchio.forwardKinematics(rmodel,rdata,q)
# M = rdata.oMi[1].copy()
# pinocchio.forwardKinematics(rmodel,rdata,q+dq)
# Md = rdata.oMi[1].copy()
# #assert( norm( M.rotation*pinocchio.exp3(w) - Md.rotation ) <1e-6 )
IDX = 13

def f(q_):
    pinocchio.forwardKinematics(rmodel,rdata,q_)
    return rdata.oMi[IDX].translation

def df_dq(q,h=1e-6):
    ndq = rmodel.nq
    dq = zero(ndq)
    f0 = f(q)
    J = zero([len(f0),ndq])
    for i in range(ndq):
        dq[i]=h
        J[:,i] = (f(q+dq)-f0)/h
        dq[i]=0
    return J

def df_dv(q,h=1e-6):
    ndq = rmodel.nv
    dq = zero(ndq)
    f0 = f(q)
    J = zero([len(f0),ndq])
    for i in range(ndq):
        dq[i]=h
        J[:,i] = (f(pinocchio.integrate(rmodel,q,dq))-f0)/h
        dq[i]=0
    return J

assert( norm( df_dv(q)[:,:3]-rdata.oMi[1].rotation) < 1e-6 )
assert( norm( df_dv(q)[:,6:]-df_dq(q)[:,7:]) < 1e-6 )

Fv= df_dv(q)[:,3:6]
Fq= df_dq(q)[:,3:7]
H = dw_dq(q[3:7])

pinocchio.computeJointJacobians(rmodel,rdata,q)
J6=pinocchio.getJointJacobian(rmodel,rdata,IDX,LOCAL)
J3=pinocchio.getJointJacobian(rmodel,rdata,IDX,LOCAL)[:3,3:6]

assert( norm( rdata.oMi[-1].rotation*J3 - Fv ) < 1e-6 )

#w = rand(3)*1e-3 
q4 = q[3:7]
dq4 = rand(4)*1e-2
dq = zero(rmodel.nq); dq[3:7] = dq4
#w = w_q(dq4)
w = dw_dq(q4)*dq4
vq = zero(rmodel.nv); vq[3:6] = w 

IDX = 1
pinocchio.forwardKinematics(rmodel,rdata,q); M = rdata.oMi[IDX].copy()
pinocchio.forwardKinematics(rmodel,rdata,q+dq); Mq = rdata.oMi[IDX].copy()
pinocchio.forwardKinematics(rmodel,rdata,pinocchio.integrate(rmodel,q,vq)); Mv = rdata.oMi[IDX].copy()

assert( norm( M.rotation*pinocchio.exp3(w) - Mv.rotation ) < 1e-6 )


# ==============================================
IDX = 1

vq2q = lambda vq: pinocchio.integrate(rmodel,rmodel.neutralConfiguration,vq)
q2vq = lambda q: pinocchio.difference(rmodel,rmodel.neutralConfiguration,q)


def f(v):
    pinocchio.forwardKinematics(rmodel,rdata,vq2q(v))
    return rdata.oMi[IDX].translation

def df_dq(v,h=1e-6):
    ndq = rmodel.nv
    dq = zero(ndq)
    f0 = f(v)
    J = zero([len(f0),ndq])
    for i in range(ndq):
        dq[i]=h
        J[:,i] = (f(v+dq)-f0)/h
        dq[i]=0
    return J

v = rand(rmodel.nv)
dv = zero(rmodel.nv)



'''
f(v) = f(q(v))
df/dv = T_q f T_v q = F_q Q_v




'''

pinocchio.computeJointJacobians(rmodel,rdata,vq2q(v))
J6=pinocchio.getJointJacobian(rmodel,rdata,IDX,LOCAL)
F_q=rdata.oMi[IDX].rotation*pinocchio.getJointJacobian(rmodel,rdata,IDX,LOCAL)[:3,:]
Q_v = pinocchio.dIntegrate(rmodel,rmodel.neutralConfiguration,v)[1]


