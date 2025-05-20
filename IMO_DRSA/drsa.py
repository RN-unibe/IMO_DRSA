import numpy as np
import types

from itertools import combinations
from functools import lru_cache

from numpy.f2py.auxfuncs import throw_error


class DRSA:
    """
    Dominance‐Based Rough Set Approach (DRSA) for multicriteria sorting.

    F = {f1,...,fn}: list of criterion functions (or column indices)
    U: set of all objects, stored as a NumPy 2D array of shape (N, n)
    d: decision attribute, stored as a 1D array of length N with values in {1...m}

    All dominance cones, approximations, and rules are built from these.
    """

    def __init__(self, X: np.ndarray=None, d: np.ndarray=None, criteria: list=None):
        """
        :param X: NumPy array with shape (N, n), each row is an object, columns are criteria evaluated on that object
        :param d: NumPy array of length N, integer‐encoded decision classes (1, ..., m)
        :param criteria: list of column indices in X to use as F={f1,...,fn}
        """
        if (X is not None) and (d is not None) and (criteria is not None):
            self.fit(X, d, criteria)


    def fit(self, T: np.ndarray, d: np.ndarray, criteria: list):
        """
        :param T: NumPy array with shape (N, n), each row is an object, columns are criteria evaluated on that object
        :param d: NumPy array of length N, integer‐encoded decision classes (1, ..., m)
        :param criteria: list of column indices in X to use as F={f1,...,fn}
        """
        self.T = T
        self.d = d
        self.criteria_F = criteria

        self.N, self.n = T.shape
        self.m = int(d.max())

        #self._sorted_idx = {i: np.argsort(self.T[:, i]) for i in self.criteria_F}




    # ------------------------------------------------------------------
    # Dominance‐cone computations
    # ------------------------------------------------------------------
    def positive_cone(self, criteria_P: tuple) -> np.ndarray:
        """
        For each object x, return a boolean mask of U indicating positive_cone(x).

        :param criteria_P: tuple of criterion indices
        :return array of shape (N, N), but stored as a boolean matrix.
        """
        # initialize all‐True mask
        mask = np.ones((self.N, self.N), dtype=bool)

        for i in criteria_P:
            # for criterion i, y dominates x if X[y,i] >= X[x,i]
            # broadcast comparison
            mask &= (self.T[:, i][:, None] >= self.T[:, i][None, :])

        return mask  # mask[y, x] == True iff y P-dominates x



    def negative_cone(self, criteria_P: tuple) -> np.ndarray:
        """
        For each object x, return a boolean mask of U indicating negative_cone(x).

        :param criteria_P: tuple of criterion indices
        :return array of shape (N, N), but stored as a boolean matrix.
        """
        mask = np.ones((self.N, self.N), dtype=bool)

        for i in criteria_P:
            mask &= (self.T[:, i][:, None] <= self.T[:, i][None, :])

        return mask  # mask[y, x] == True iff x P-dominates y



    # ------------------------------------------------------------------
    # Rough approximations
    # ------------------------------------------------------------------
    def lower_approx_up(self, criteria_P: tuple, t: int) -> np.ndarray:
        """
        Compute the lower approximation of the 'up' region at threshold t under criteria P.

        :param criteria_P: Tuple of parameters defining the positive cone relation.
        :param t: Integer threshold for the upward approximation.
        :return: 1D boolean numpy array of length N where each entry x is True if,
                 for every y in the positive cone of x, d[y] >= t holds.
        """
        mask = self.positive_cone(criteria_P)  # shape (N,N), mask[y,x]=True iff in positive_cone(x)

        ok = np.all(~mask | (self.d[:, None] >= t), axis=0)

        return ok


    def upper_approx_up(self, criteria_P: tuple, t: int) -> np.ndarray:
        """
        Compute the upper approximation of the 'up' region at threshold t under criteria P.

        :param criteria_P: Tuple of parameters defining the negative cone relation.
        :param t: Integer threshold for the upward approximation.
        :return: 1D boolean numpy array of length N where each entry x is True if
                 there exists at least one y in the negative cone of x such that d[y] >= t.
        """
        mask = self.negative_cone(criteria_P)  # shape (N,N), mask[y,x]=True iff in negative_coneP(x)
        ok = np.any(mask & (self.d[:, None] >= t), axis=0)

        return ok


    def lower_approx_down(self, criteria_P: tuple, t: int) -> np.ndarray:
        """
        Compute the lower approximation of the 'down' region at threshold t under criteria P.

        :param criteria_P: Tuple of parameters defining the negative cone relation.
        :param t: Integer threshold for the downward approximation.
        :return: 1D boolean numpy array of length N where each entry x is True if,
                 for every y in the negative cone of x, d[y] <= t holds.
        """
        mask = self.negative_cone(criteria_P)
        ok = np.all(~mask | (self.d[:, None] <= t), axis=0)

        return ok


    def upper_approx_down(self, criteria_P: tuple, t: int) -> np.ndarray:
        """
        Compute the upper approximation of the 'down' region at threshold t under criteria P.

        :param criteria_P: Tuple of parameters defining the positive cone relation.
        :param t: Integer threshold for the downward approximation.
        :return: 1D boolean numpy array of length N where each entry x is True if
                 there exists at least one y in the positive cone of x such that d[y] <= t.
        """
        mask = self.positive_cone(criteria_P)
        ok = np.any(mask & (self.d[:, None] <= t), axis=0)

        return ok


    # ------------------------------------------------------------------
    # Quality of approximation gamma_P(Cl) (Definition §5.2)
    # ------------------------------------------------------------------
    def quality(self, criteria_P: tuple) -> float:
        """
        gamma_P(Cl) = proportion of P‐consistent objects.

        :param criteria_P: Tuple of parameters defining the positive cone relation.
        :return proportion of P‐consistent objects
        """
        consistent = np.ones(self.N, dtype=bool)
        # a sample union: Cl^{>=2},...,Cl^{>=m} boundaries

        for t in range(2, self.m + 1):
            lower = self.lower_approx_up(criteria_P, t)
            upper = self.upper_approx_up(criteria_P, t)
            boundary = upper & ~lower
            consistent &= ~boundary

        return consistent.sum() / self.N



    # ------------------------------------------------------------------
    # Finding reducts (brute‐force; not good for large n, use heuristic)
    # ------------------------------------------------------------------
    def find_reducts(self):
        """
        Return all minimal criteria_P subset of criteria_F such that gamma_P = gamma_F.
        TODO: combinatorial. Use only for small n or with pruning.
        """
        full_gamma = self.quality(tuple(self.criteria_F))
        reducts = []

        for r in range(1, len(self.criteria_F) + 1):

            for criteria_P in combinations(self.criteria_F, r):
                if self.quality(criteria_P) == full_gamma:
                    # minimality: no subset of criteria_P already in reducts
                    if not any(set(R).issubset(criteria_P) for R in reducts):
                        reducts.append(criteria_P)

            if reducts:
                break

        return reducts



    # ------------------------------------------------------------------
    # Decision‐rule induction
    # ------------------------------------------------------------------
    def induce_rules(self, criteria_P: tuple, union_type='up', t=None):
        """
        Induce certain / possible decision rules for Cl^{>=t} or Cl^{≤t}.
        union_type: 'up' or 'down'
        t: class index
        Returns list of rules of form (conditions, conclusion, support, confidence)
        """
        rules = []

        # Collect candidate profiles from either lower or upper approximation
        if union_type == 'up':
            lower = self.lower_approx_up(criteria_P, t)
            upper = self.upper_approx_up(criteria_P, t)
            bases = np.where(lower)[0]
            possibles = np.where(upper & ~lower)[0]

            # Build rules “if fi(x) >= ri for i in P then x in Cl^{>=t}” from bases
            for idx in bases:
                profile = {i: self.T[idx, i] for i in criteria_P}

                # compute support & confidence
                mask = np.ones(self.N, dtype=bool) # Cn set
                for i, fx in profile.items():
                    mask &= (self.T[:, i] >= fx)

                support = mask.sum() / self.N
                confidence = (self.d[mask] >= t).mean()

                rules.append((profile, f'd >= {t}', support, confidence, 'certain'))
            
            
            for idx in possibles:
                profile = {i: self.T[idx, i] for i in criteria_P}

                mask = np.ones(self.N, dtype=bool)

                for i, fx in profile.items():
                    mask &= (self.T[:, i] >= fx)

                support = mask.sum() / self.N
                confidence = (self.d[mask] >= t).mean()

                rules.append((profile, f'd >= {t}', support, confidence, 'possible'))
            
        elif union_type == 'down':
            lower = self.lower_approx_down(criteria_P, t)
            upper = self.upper_approx_down(criteria_P, t)
            bases = np.where(lower)[0]
            possibles = np.where(upper & ~lower)[0]

            # Build rules “if fi(x) <= ri for i in P then x in Cl^{<=t}” from bases
            for idx in bases:
                profile = {i: self.T[idx, i] for i in criteria_P}

                # compute support & confidence
                mask = np.ones(self.N, dtype=bool)  # Cn set
                for i, fx in profile.items():
                    mask &= (self.T[:, i] <= fx)

                support = mask.sum() / self.N
                confidence = (self.d[mask] <= t).mean()

                rules.append((profile, f'd <= {t}', support, confidence, 'certain'))

            for idx in possibles:
                profile = {i: self.T[idx, i] for i in criteria_P}

                mask = np.ones(self.N, dtype=bool)

                for i, fx in profile.items():
                    mask &= (self.T[:, i] <= fx)

                support = mask.sum() / self.N
                confidence = (self.d[mask] <= t).mean()

                rules.append((profile, f'd <= {t}', support, confidence, 'possible'))

        else:
            throw_error('Invalid union_type')

        return rules




    def explain_rules(self, rules, verbose:bool=False):
        """
        Just write them into If-then statements.

        :param rules:
        :param verbose:
        :return:
        """
        explain = []
        for cond, concl, support, conf, kind in rules:
            cond_str = " AND ".join(f"f_{i + 1} >= {v}" for i, v in cond.items())
            rule_string = f"[{kind.upper()}] IF {cond_str} THEN {concl}  (support={support:.2f}, confidence={conf:.2f})"
            explain.append(rule_string)

            if verbose:
                print(rule_string)

        return explain


    #def run(self):

    # ------------------------------------------------------------------
    # Getters and Setters
    # ------------------------------------------------------------------












