forces = pinocchio.StdVect_Force()
for i in range(rmodel.njoints): forces.append(pinocchio.Force.Zero())

tau,fr,fl = pbm.x2vars(res)

Mr = rmodel.frames[pbm.idR].placement
jr = rmodel.frames[pbm.idR].parent
forces[jr] = Mr.act(pinocchio.Force(fr))

Ml = rmodel.frames[pbm.idL].placement
jl = rmodel.frames[pbm.idL].parent
fl = pinocchio.Force(fl)
forces[jl] = Mr.act(pinocchio.Force(fl))

print(pinocchio.rnea(rmodel,rdata,pbm.q,pbm.vq,zero(rmodel.nv),forces)-tau)
