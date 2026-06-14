from dso.core import DeepSymbolicOptimizer

try:
    from dso.task.regression.sklearn import DeepSymbolicRegressor
except Exception:
    DeepSymbolicRegressor = None
