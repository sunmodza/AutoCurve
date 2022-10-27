import numpy as np

import AutoCurve

v = np.linspace(0, 10, 100)
m = np.linspace(0.1, 3, 100)
ele = {"V": v}
ans = v**m+3


tool = AutoCurve.IQ180_Solution()
tool.set_start(ele, ans, sol_nums=1, not_use_all=False,
               allow_fac=False,  prob_of_use_x=50,reset_params = 500,keep_factor = 2,keep_best_variable_params=1)
a = tool.run_a_solution(accept_error=0.00003)


print(a.eval_eq({"M": 5, "V": 2}))
print(tool.worked_text)


