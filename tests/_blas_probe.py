import sys
print("py", sys.version, flush=True)
import numpy as np
print("numpy", np.__version__, flush=True)
a = np.array([[1.0, 0.0], [0.0, 1.0]])
b = np.array([[2.0, 0.0], [0.0, 3.0]])
print("add", a + b, flush=True)
print("dot", np.dot(a, b), flush=True)
print("done", flush=True)
