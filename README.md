 # Interactive Optimisation Process using DRSA (IMO-DRSA)
The method consists of an iterative cycle of two main stages:

## A. Optimisation Stage
Generate a sample from the current approximation of the Pareto optimal set.

## B. Dialogue Stage
- The DM evaluates the sample and selects "good" solutions.

- DRSA is applied to infer decision rules distinguishing "good" from "others."

- These rules are logical constraints used to narrow the search space.

- The process repeats with a refined Pareto sample until a satisfactory solution is found.


## Step-by-step Overview:
1. Optimize: get Pareto front

2. Ask DM to sort into "good" / "other"

3. Use DRSA:
   - Compute approximations
   - Drive rules 
   - Pick best ones (maybe with DM support)

4. Generate new constraints

5. Feed constraints to DRSAProblem

6. Repeat



## Sources:

- Branke, J., Deb, K., Miettinen, K., & Słowiński, R. (Eds.). (2008). Multiobjective optimization: Interactive and evolutionary approaches (Vol. 5252). Springer. https://doi.org/10.1007/978-3-540-88908-3
- Dominance-based Rough Set Approach (DRSA): https://en.wikipedia.org/wiki/Dominance-based_rough_set_approach
- Rough Set Theory: https://en.wikipedia.org/wiki/Rough_set
- Multi-Criteria Decision Analysis (MCDA): https://en.wikipedia.org/wiki/Multi-criteria_decision_analysis
- pymoo (https://pymoo.org/)


## License

This project uses the following third-party packages, which are licensed under the MIT License:
- pymoo (https://pymoo.org/) - MIT License


## Notes

- The term 'rule' and 'constraint' are used interchangeably.