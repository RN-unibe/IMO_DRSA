import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
np.row_stack = np.vstack # Because pymoo uses np.vstack which is deprecated

from typing import Callable, List

import pandas as pd
from matplotlib import pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from src.imo_drsa.decision_maker import BaseDM, InteractiveDM
from src.imo_drsa.drsa import DRSA
from src.imo_drsa.problem_extender import ProblemExtender


class IMO_DRSAEngine():
    """
    Interactive Multi-objective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

    :param universe: Initial decision variable matrix of shape (n_solutions, n_variables).
    :param objectives: List of objective functions mapping a solution vector to a float.
    """

    def __init__(self):  # , algorithm=NSGA2, **kwargs):
        """
        Initialise the DRSA and NSGA2 Classes.

        :param algorithm: Algorithm to use. Must be pymoo compatible
        :param kwargs: any and all algorithm parameters
        """
        self.history = []
        self.drsa = DRSA()
        self.wrapper = ProblemExtender()
        self.pareto_front = None
        self.pareto_set = None
        self.rules = None

    def fit(self, problem: Problem, objectives: List[Callable] = None, verbose: bool = False, to_file=True,
            pymoo_verbose=False):
        """
        Fit the IMO-DRSA solver.

        :param problem: Problem to be optimised
        :param objectives: List of objective functions mapping a solution vector to a float.
        :param verbose: bool if graphs and printouts should be given
        :param to_file: bool if output should be saved to file
        :param pymoo_verbose: bool if pymoo notifications should be allowed
        :return:
        """
        self.problem = self.wrapper.enable_dynamic_constraints(problem=problem)
        self.objectives = objectives
        self.verbose = verbose
        self.pymoo_verbose = pymoo_verbose
        self.pareto_front, self.pareto_set = None, None

        self.to_file = to_file

        if to_file:
            # Setup results directory with timestamp
            now = datetime.now()
            self.ts = now.strftime("%Y%m%d_%H%M%S")
            self.out_dir = Path("../../results") / f"results_{self.ts}"
            self.out_dir.mkdir(parents=True, exist_ok=True)

        return self

    def run(self, decision_maker: BaseDM, visualise: bool = False, max_iter: int = 5) -> bool:
        """
        Run the interactive optimisation loop.

        :param decision_maker: Interface for classification and feedback.
        :param visualise: Whether to plot the Pareto front each iteration.
        :param max_iter: Maximum interactive iterations.
        :return: True if session finishes successfully, False otherwise.
        """

        P_idx = tuple([i for i in range(0, len(self.objectives))])
        decision_attribute = None
        pareto_front_sample, pareto_set_sample = self.calculate_pareto_front(pop_size=100)

        iteration: int = 0
        while iteration < max_iter:

            print(len(self.pareto_front))

            state_backup = {
                'problem': deepcopy(self.problem),
                'pareto_front': deepcopy(self.pareto_front),
                'pareto_set': deepcopy(self.pareto_front),
                'rules': deepcopy(self.rules)
            }
            
            self.history.append(state_backup)

            if visualise:
                self.visualise2D(pareto_front_sample, pareto_set_sample, iter=iteration, nr=1)

            self.drsa.fit(pareto_set=pareto_set_sample, criteria=P_idx, decision_attribute=decision_attribute)

            # Induce association rules from current table
            if decision_maker.is_interactive():
                association_rules = self.drsa.find_association_rules(min_support=0.2, min_confidence=0.9)
                _, assoc_summary = self.drsa.summarize_association_rules(association_rules)

            else:
                assoc_summary = ""

            # Classify with DM feedback
            decisions = decision_maker.classify(T=pareto_set_sample, X=pareto_front_sample, assoc_rules_summary=assoc_summary)

            if 2 not in decisions: # No samples were considered 'good', i.e., none are in class 2
                print("Trying again...")
                pareto_front_sample, pareto_set_sample = self.calculate_pareto_front()
                continue

            # Find a reduct and induce decision rules
            self.drsa.fit(pareto_set=pareto_set_sample, criteria=P_idx, decision_attribute=decisions)

            reducts = self.drsa.find_reducts()
            core = self.drsa.core(reducts)

            reduct = decision_maker.select_reduct(reducts, core)

            self.rules = self.drsa.induce_decision_rules(reduct, minimal=False)

            # DM selects preferred rules
            selected = decision_maker.select(self.rules)
            # print(selected)

            if len(selected) == 0:
                print("Trying again...")
                pareto_front_sample, pareto_set_sample = self.calculate_pareto_front()
                continue

            # Generate new constraints from selected rules
            new_constraints = self.generate_constraints(selected)
            self.problem.add_constraints(new_constraints)

            if visualise:
                self.visualise2D(pareto_front_sample, pareto_set_sample, iter=iteration, nr=2)

            # Compute Pareto front under current constraints
            pareto_front_sample, pareto_set_sample = self.calculate_pareto_front()


            if decision_maker.is_interactive():
                undo = input("Do you want to undo the last selection? (y/n): ")

                if undo.lower() == 'y':
                    pareto_front_sample, pareto_set_sample, rules = self.undo_last()
                    continue

            # Ask DM if current solutions are satisfactory
            if decision_maker.is_satisfied(pareto_front_sample, pareto_set_sample, self.rules):
                print("DM is satisfied. Terminating Process")


                if self.to_file:
                    self.write_to_file()

                return True

            # If the constraints create a solely infeasible- or empty region, the last selection is undone
            if pareto_front_sample is None or pareto_front_sample.size == 0:
                print("Infeasible constraints: please revise.")

                pareto_front_sample, pareto_set_sample, self.rules = self.undo_last()

                continue

            iteration += 1

        print('Maximum iterations reached.')
        print('Terminating now.')


        return False

    def undo_last(self):
        """
        Undoes the last iteration.

        :return: the last state of pareto_front, pareto_set, and rules
        """

        last_state = self.history.pop()
        self.problem = last_state['problem']
        self.pareto_front = last_state['pareto_front']
        self.pareto_set = last_state['pareto_set']
        self.rules = last_state['rules']

        print("Last selection undone.")

        return self.pareto_front, self.pareto_set, self.rules

    def visualise2D(self, new_pareto_front=None, new_pareto_set=None, all_kwargs=None, sub_kwargs=None,
                    title_front=None, xlabel_front=None, ylabel_front=None, title_set=None, xlabel_set=None,
                    ylabel_set=None, legend=True, iter=0, nr=1):
        """
        Plot both the Pareto front in decision space and the Pareto set in objective space,
        showing the original (lightgrey) vs. the new subset (red).

        :param iter:
        :param new_pareto_set: Current Pareto set (objectives), shape (n_points, 2).
        :param new_pareto_front: Current Pareto front (decision vars), shape (n_points, 2).
        :param all_kwargs: Style overrides for 'all points'.
        :param sub_kwargs: Style overrides for the highlighted subset.
        :param title_front: Title for the decision-variable plot.
        :param xlabel_front: X-label for the decision-variable plot.
        :param ylabel_front: Y-label for the decision-variable plot.
        :param title_set: Title for the objective-space plot.
        :param xlabel_set: X-label for the objective-space plot.
        :param ylabel_set: Y-label for the objective-space plot.
        :param legend: Whether to show legends.
        """

        all_kwargs = {} if all_kwargs is None else all_kwargs
        sub_kwargs = {} if sub_kwargs is None else sub_kwargs

        all_defaults = dict(color='lightgrey', marker='o', label='all points')
        sub_defaults = dict(color='red', marker='x', s=100, label='subset')

        all_style = {**all_defaults, **all_kwargs}
        sub_style = {**sub_defaults, **sub_kwargs}

        if new_pareto_front is None:
            new_pareto_front = self.pareto_front

        if new_pareto_set is None:
            new_pareto_set = self.pareto_set

        fig, (ax_front, ax_set) = plt.subplots(1, 2, figsize=(12, 5))

        # --- decision-space plot ---
        front = np.asarray(self.pareto_front)
        ax_front.scatter(front[:, 0], front[:, 1], **all_style)

        new_front = np.asarray(new_pareto_front)
        ax_front.scatter(new_front[:, 0], new_front[:, 1], **sub_style)

        ax_front.set_title(title_front or "Pareto Front (decision vars)")
        if xlabel_front:
            ax_front.set_xlabel(xlabel_front)
        if ylabel_front:
            ax_front.set_ylabel(ylabel_front)
        if legend:
            ax_front.legend()

        # --- objective-space plot ---
        pset = np.asarray(self.pareto_set)
        ax_set.scatter(pset[:, 0], pset[:, 1], **all_style)

        new_pset = np.asarray(new_pareto_set)
        ax_set.scatter(new_pset[:, 0], new_pset[:, 1], **sub_style)

        ax_set.set_title(title_set or "Pareto Set (objectives)")
        if xlabel_set:
            ax_set.set_xlabel(xlabel_set)
        if ylabel_set:
            ax_set.set_ylabel(ylabel_set)
        if legend:
            ax_set.legend()

        plt.tight_layout()

        if self.to_file:
            out_dir = self.out_dir
            plt.savefig(f"{out_dir}/graph_{iter}.{nr}.png", format='png')

        if self.verbose:
            plt.show()



    def calculate_pareto_front(self, n_gen=100, pop_size=100, sample_size=10) -> (np.ndarray, np.ndarray):
        """
        Compute Pareto-optimal set using NSGA2 algorithm.

        :param sample_size:
        :param pop_size:
        :param pop_size: Population size for NSGA2.
        :param n_gen: Number of generations.

        :return: Tuple of decision variables (pareto_front) and objective values of Pareto front (pareto_set).
        """
        pop_size = pop_size if pop_size >= sample_size else 10
        res = minimize(self.problem, NSGA2(pop_size=pop_size), termination=('n_gen', n_gen), verbose=self.pymoo_verbose)
        self.pareto_front, self.pareto_set = res.X, res.F

        sample = min(len(self.pareto_front), sample_size)
        idx = np.random.choice(len(self.pareto_front), sample, replace=False)

        pareto_front_sample = self.pareto_front[idx]
        pareto_set_sample = self.pareto_set[idx]

        return pareto_front_sample, pareto_set_sample

    def generate_constraints(self, selected_rules, elementwise=None) -> List[Callable]:
        """
        Translate selected decision rules into inequality constraints g(x) <= 0.

        :param selected_rules: List of decision rules (profile, conclusion, support, confidence, kind, direction, description).
        :return: Constraints as functions mapping x to a float (<=0).
        """
        constraints = []

        if elementwise is None:
            elementwise = self.problem.elementwise

        for profile, _, _, _, _, _, desc in selected_rules:

            for idx, threshold in profile.items():
                if elementwise:
                    constraints.append(lambda x, i=idx, th=threshold: self.objectives[i](x) - th)
                else:
                    constraints.append(
                        lambda X, i=idx, th=threshold: np.array([self.objectives[i](xi) for xi in X]) - th)

        return constraints


    def write_to_file(self):
        """
        Write the found pareto_front, pareto_set, and the found rules to file.
        """
        out_dir = self.out_dir

        pf_path = out_dir / f"pareto_front.csv"
        ps_path = out_dir / f"pareto_set.csv"
        rules_path = out_dir / f"rules.json"

        if hasattr(self.pareto_front, 'shape'):
            df_pf = pd.DataFrame(self.pareto_front,
                                 columns=[f"x{i}" for i in range(self.pareto_front.shape[1])])
            df_pf.to_csv(pf_path, index=False)

        if hasattr(self.pareto_set, 'shape'):
            df_ps = pd.DataFrame(self.pareto_set,
                                 columns=[f"f{i}" for i in range(self.pareto_set.shape[1])])
            df_ps.to_csv(ps_path, index=False)

        serializable = []
        for rule in self.rules:
            profile, conclusion, support, confidence, kind, direction, description = rule
            serializable.append({
                "profile": profile,
                "conclusion": conclusion,
                "support": support,
                "confidence": confidence,
                "kind": kind,
                "direction": direction,
                "description": description
            })
        with open(rules_path, 'w') as f:
            json.dump(serializable, f, indent=4)

        if getattr(self, 'verbose', False):
            print(f"Saved Pareto front to: {pf_path}")
            print(f"Saved Pareto set to: {ps_path}")
            print(f"Saved rules to: {rules_path}")