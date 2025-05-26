# Interactive Multi-Objective Optimisation with DRSA (IMO-DRSA)

A Python implementation of the Interactive Multi-Objective Optimisation framework using the Dominance-Based Rough Set Approach (IMO-DRSA). 
This interactive and iterative optimisation process integrates human decision-making preferences via DRSA rule induction, continually refining the Pareto-optimal solutions.

## Overview

### Framework Components

- **IMO-DRSA Engine** (`engine.py`):
  - Manages iterative optimisation loops (using NSGA-II to calculate Pareto fronts).
  - Visualises Pareto fronts and objective spaces interactively.
  - Generates new constraints dynamically based on decision-maker selections.


- **Dominance-Based Rough Set Approach (DRSA)** (`drsa.py`):
  - Induces decision rules distinguishing preferred solutions ('good') from non-preferred ones.
  - Computes positive/negative cones, rough approximations, and quality metrics.
  - Supports association rule mining for enhanced decision-context understanding.


- **Decision Makers** (`decision_maker.py`):
  - **InteractiveDM**: Integrates human input to classify Pareto samples and select DRSA rules.
  - **AutomatedDM**: Automatically classifies, selects rules, and checks convergence criteria.
  - **DummyDM**: Provides trivial decision-making logic for unit tests.


- **Problem Extender** (`problem_extender.py`):
  - Dynamically adds inequality constraints to pymoo optimisation problems.
  - Compatible with both elementwise and batch evaluations.


## Iterative Optimisation Cycle

1. **Optimisation Stage**: Generate a Pareto-optimal solution set (via NSGA-II).
2. **Dialogue Stage**:
   - Decision-maker classifies solutions ('good'/'other').
   - DRSA infers decision rules based on classified solutions.
   - Decision-maker selects the most relevant rules.
   - Rules become constraints to refine the Pareto search space.

The cycle continues until a satisfactory solution set emerges or convergence criteria are met.


## Setup
Run `setup.sh` on MacOS/Linux or `setup.ps1` on Windows.

## Sources

- Branke, J., Deb, K., Miettinen, K., & Słowiński, R. (2008). *Multiobjective optimisation: Interactive and evolutionary approaches*. Springer. [Link](https://doi.org/10.1007/978-3-540-88908-3)
- [Dominance-based Rough Set Approach (DRSA)](https://en.wikipedia.org/wiki/Dominance-based_rough_set_approach)
- [Rough Set Theory](https://en.wikipedia.org/wiki/Rough_set)
- [Multi-Criteria Decision Analysis (MCDA)](https://en.wikipedia.org/wiki/Multi-criteria_decision_analysis)
- [pymoo Optimisation Library](https://pymoo.org/)

## License

This project includes third-party software licensed under the MIT License:
- [pymoo](https://pymoo.org/) - MIT License

