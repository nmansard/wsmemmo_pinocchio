refQ = robot.q0

def cost(q):
    residuals = m2a(q-refQ)
    return .5*sum(residuals**2)

def dCost(q):
    dq = m2a(q-refQ)
    return dq
  
def numdiffCost(q,h=1e-6):
    f0 = cost(q)
    nq,nf = len(q),1
    dq = zero(nq)
    df_dq = zero([nf,nq])
    for i in range(nq):
        dq[i] = h
        df_dq[:,i] = (cost(q+dq)-f0)/h
        dq[i] = 0
    return df_dq

q=rand(robot.model.nq)
norm(dCost(q)-numdiffCost(q))
