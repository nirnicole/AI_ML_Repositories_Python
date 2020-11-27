from csp import Constraint

"""
    Summery: Crossword problem,
             This class contains relevant Crossword constrains.
    Author:  Nir Nicole
"""

"""A1 constrains"""


class A1D1Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A1D1Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # first constrain: index char must not be uniqe
        return assignment[self.variable_1][0] == assignment[self.variable_2][0]


class A1D2Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A1D2Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # second constrain: second char in a1 must be first in d2
        return assignment[self.variable_1][1] == assignment[self.variable_2][0]


class A1D3Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A1D3Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # third constrain: third char in a1 must be first in d3
        return assignment[self.variable_1][2] == assignment[self.variable_2][0]


""" A2 constrains"""


class A2D1Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A2D1Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # first constrain: first char in a1 must be second d1
        return assignment[self.variable_1][0] == assignment[self.variable_2][1]


class A2D2Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A2D2Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # second constrain: second char in a1 must be same in d2
        return assignment[self.variable_1][1] == assignment[self.variable_2][1]


class A2D3Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A2D3Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # third constrain: third char in a2 must be second in d3
        return assignment[self.variable_1][2] == assignment[self.variable_2][1]


""" A3 constrains"""


class A3D1Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A3D1Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # first constrain: first char in a3 must be kast in d1
        return assignment[self.variable_1][0] == assignment[self.variable_2][2]


class A3D2Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A3D2Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # second constrain: second char in a1 must be last in d3
        return assignment[self.variable_1][1] == assignment[self.variable_2][2]


class A3D3Constraint(Constraint):
    def __init__(self, variable_1, variable_2):
        # variable_1: str, variable_2: str) -> None
        super(A3D3Constraint, self).__init__([variable_1, variable_2])
        self.variable_1 = variable_1
        self.variable_2 = variable_2

    def satisfied(self, assignment):
        if self.variable_1 not in assignment or self.variable_2 not in assignment:
            return True

        # third constrain: third char need to be the same in d3
        return assignment[self.variable_1][2] == assignment[self.variable_2][2]
