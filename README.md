 # Interactive Optimisation Process using DRSA (IMO-DRSA)
The method consists of an iterative cycle of two main stages:

## A. Calculation Stage
Generate a sample from the current approximation of the Pareto optimal set.

## B. Dialogue Stage
- The DM evaluates the sample and selects "good" solutions.

- DRSA is applied to infer decision rules distinguishing "good" from "others."

- These rules are logical constraints used to narrow the search space.

- The process repeats with a refined Pareto sample until a satisfactory solution is found.

## Step-by-step Overview:
1. Sample solutions from the Pareto set.

2. Present sample + derived association rules to the DM.

3. If satisfied, stop; otherwise:

4. DM marks "good" solutions.

5. DRSA derives rules like: "$f_1(x) \leq \alpha_1 \quad \text{and} \quad f_2(x) \leq \alpha_2, then x is good."$

6. Rules are reviewed and selected by the DM.

7. Selected rules are turned into new constraints.

8. Return to Step 1 with updated constraints.

Repeat until satisfactory solution is found.



## Sources:

- Branke, J., Deb, K., Miettinen, K., & Słowiński, R. (Eds.). (2008). Multiobjective optimization: Interactive and evolutionary approaches (Vol. 5252). Springer. https://doi.org/10.1007/978-3-540-88908-3
- Dominance-based Rough Set Approach (DRSA): https://en.wikipedia.org/wiki/Dominance-based_rough_set_approach
- Rough Set Theory: https://en.wikipedia.org/wiki/Rough_set
- Multi-Criteria Decision Analysis (MCDA): https://en.wikipedia.org/wiki/Multi-criteria_decision_analysis


## License

This project uses the following third-party packages, which are licensed under the MIT License:
- pymoo (https://pymoo.org/) - MIT License
