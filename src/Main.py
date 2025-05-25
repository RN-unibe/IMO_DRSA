from imo_drsa.decision_maker import InteractiveDM
from imo_drsa.engine import IMO_DRSAEngine

from pymoo.problems import get_problem

if __name__ == "__main__":
    dm = InteractiveDM()
    problem = get_problem("ackley")

    engine = IMO_DRSAEngine()

    engine.fit(problem=problem)

    engine.solve(dm)
