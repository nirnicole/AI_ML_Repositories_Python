from csp import Constraint

"""
    Summery: Map coloring problem,
             This class contains relevant coloring constrains.
    Author:  Nir Nicole
"""

"""Map coloring constrains"""

# implementing framework constrain.
class MapColoringConstraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(MapColoringConstraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    # if not yet to be assigned they are not conflicting for sure.
    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # check if 2 adjust colors are not conflicting(not the same)
        return assignment[self.variable_1] != assignment[self.variable_2]

