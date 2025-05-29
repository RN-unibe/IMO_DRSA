from imo_drsa.decision_maker import InteractiveDM
from imo_drsa.engine import IMO_DRSAEngine

from pymoo.problems import get_problem

if __name__ == "__main__":
    dm = InteractiveDM()

    problem = get_problem("bnh")


    def f0(x):
        return 4 * x[0] * x[0] + 4 * x[1] * x[1]


    def f1(x):
        term1 = x[0] - 5
        term2 = x[1] - 5

        return term1 * term1 + term2 * term2


    objectives = [f0, f1]

    engine = IMO_DRSAEngine().fit(problem=problem, objectives=objectives, verbose=True)

    success = engine.run(dm, max_iter=4)
