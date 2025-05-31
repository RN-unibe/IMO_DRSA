from imo_drsa.decision_maker import InteractiveDM
from imo_drsa.engine import IMO_DRSAEngine

from pymoo.problems import get_problem

if __name__ == "__main__":
    # 1. Create an Interactive Decision Maker
    dm = InteractiveDM()

    # 2. Instantiate your problem. This must be pymoo compatible.
    problem = get_problem("bnh")

    # 3. Define the objectives explicitly as Python callables.
    def f0(x):
        return 4 * x[0] * x[0] + 4 * x[1] * x[1]

    def f1(x):
        term1 = x[0] - 5
        term2 = x[1] - 5

        return term1 * term1 + term2 * term2

    objectives = [f0, f1]

    # 4. Fit the IMO-DRSA Engine with the chosen problem and objectives
    engine = IMO_DRSAEngine().fit(problem=problem, objectives=objectives,
                                  verbose=True,   # Set Ture to be given print out updates
                                  visualise=True, # Set True to be given 2D plots of the current pareto fronts and sets
                                  to_file=True)   # Set True to save outputs to /results_YYYYMMDD_HHMMSS/)

    # 5. Run the engine
    success = engine.run(dm, max_iter=4)
