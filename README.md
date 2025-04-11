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
1. Generate a representative sample of solutions from the currently considered part of the Pareto optimal set. 
2. Present the sample to the DM, possibly together with association rules showing relationships between attainable values of objective functions in the Pareto optimal set. 
3. If the DM is satisfied with one solution from the sample, then this is the most preferred solution and the procedure stops. Otherwise continue. 
4. Ask the DM to indicate a subset of “good” solutions in the sample. 
5. Apply DRSA to the current sample of solutions sorted into “good” and “others”, in order to induce a set of decision rules with the following syntax “if f_j1 (x) ≤ α_j1 and ... and f_jp (x) ≤ α_jp , then solution x is good”, {j_1,...,j_p} ⊆ {1,...,n}.  
6. Present the obtained set of rules to the DM.  
7. Ask the DM to select the decision rules most adequate to his/her preferences.
8. Adjoin the constraints f_j1 (x) ≤ α_j1 , ... , f_jp (x) ≤ α_jp coming from the rules selected in Step 7 to the set of constraints imposed on the Pareto optimal set, in order to focus on a part interesting from the point of view of DM’s preferences.
9. Go back to Step 1.



## Sources:

- Branke, J., Deb, K., Miettinen, K., & Słowiński, R. (Eds.). (2008). Multiobjective optimization: Interactive and evolutionary approaches (Vol. 5252). Springer. https://doi.org/10.1007/978-3-540-88908-3
- Dominance-based Rough Set Approach (DRSA): https://en.wikipedia.org/wiki/Dominance-based_rough_set_approach
- Rough Set Theory: https://en.wikipedia.org/wiki/Rough_set
- Multi-Criteria Decision Analysis (MCDA): https://en.wikipedia.org/wiki/Multi-criteria_decision_analysis
- pymoo (https://pymoo.org/)


## License

This project uses the following third-party packages, which are licensed under the MIT License:
- pymoo (https://pymoo.org/) - MIT License
