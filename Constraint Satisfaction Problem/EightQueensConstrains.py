from csp import Constraint

"""
    Summery: The eight queens problem problem,
             This class contains relevant queens constrains.
    Author:  Nir Nicole
"""

"""The eight queens constrains"""

# implementing framework constrain.
class EightQueensConstraint(Constraint):
    def __init__(self, columns):
        super(EightQueensConstraint, self).__init__(columns)
        self.columns = columns

    def satisfied(self, assignment):
        for q1c, q1r in assignment.items():  # q1c = queen 1 column, q1r = queen 1 row
            for q2c in range(q1c + 1, len(self.columns) + 1):  # q2c = queen 2 column
                if q2c in assignment:
                    q2r = assignment[q2c]  # q2r = queen 2 row
                    if q1r == q2r:  # same row?
                        return False
                    if abs(q1r - q2r) == abs(q1c - q2c):  # same diagonal?
                        return False
        return True  # no conflict

    def Conflicts(self, assignment, variable):

        violations_count = 0
        q1c, q1r = variable, assignment[variable]  # q1c = queen 1 column, q1r = queen 1 row
        for q2c in range(0, len(self.columns)):  # q2c = queen 2 column
            if q2c != q1c:
                if q2c in assignment:
                    q2r = assignment[q2c]  # q2r = queen 2 row
                    if q1r == q2r:  # same row?
                        violations_count = violations_count + 1
                    if abs(q1r - q2r) == abs(q1c - q2c):  # same diagonal?
                        violations_count = violations_count + 1
        return violations_count
