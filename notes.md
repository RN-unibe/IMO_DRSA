In pymoo:
- xl: Float or np.ndarray of length n_var representing the lower bounds of the design variables.

- xu: Float or np.ndarray of length n_var representing the upper bounds of the design variables.

are the lower and upper bounds we want to find with DRSA, I think.















    def certainty(self, X=None) -> float :
        """
        "The difference between the upper and lower approximation constitutes
        the boundary region of the rough set, whose elements cannot be characterized
        with certainty as belonging or not to X (by using the available information).
        The information about objects from the boundary region is, therefore, inconsistent.
        The cardinality of the boundary region states, moreover, the extent
        to which it is possible to express X in terms of certainty, on the basis of the
        available information. In fact, these objects have the same description, but are
        assigned to different classes, such as patients having the same symptoms (the
        same description), but different pathologies (different classes). For this reason,
        this cardinality may be used as a measure of inconsistency of the information
        about X." (p. 125 bzw. 141)


        """


        #Do a thing with the cardinality of X, I guess
        pass