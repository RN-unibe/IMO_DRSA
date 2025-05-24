import numpy as np

from imo_drsa.imo_drsa import IMO_DRSA
from imo_drsa.problem import DRSABaseProblem
from src.imo_drsa.decision_maker import InteractiveDM

if __name__ == "__main__":
    dm = InteractiveDM()
    engine = IMO_DRSA()

    engine.fit()

    engine.solve(dm)
