def constraint_quaternion(x):
    return norm(x[3:7])-1

result = fmin_slsqp(x0       = x0,
                    func     = pbm.cost,
                    f_eqcons = constraint_quaternion,
                    callback = pbm.callback)
print(result[3:7])
