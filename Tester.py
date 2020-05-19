from random import randrange

from EightQueensConstrains import *
from MapColoringConstrains import *
from CrosswordConstrains import *
import csp
import timeit

"""
    Summery: Tester for various types of csp problems
    Author:  Nir Nicole
"""

"""-------------------------------------------------------------------------"""
"""Crossword puzzle test:"""


def testCrossword():
    print "\nTesting Crossword puzzle:\n"
    start_time = timeit.default_timer()

    variables = ["A1", "A2", "A3", "D1", "D2", "D3"]
    domains = {}
    for variable in variables:
        domains[variable] = ["add", "ado", "age", "ago", "aid", "ail", "aim", "air", "and", "any", "ape", "apt", "arc",
                             "are", "ark", "arm", "art", "ash", "ask", "auk", "awe", "awl", "aye", "bad", "bag", "ban",
                             "bat", "bee",
                             "boa", "ear", "eel", "eft", "far", "fat", "fit", "lee", "oaf", "rat", "tar", "tie"]

    csp_instance = csp.CSP(variables, domains, {"MRV": True, "LCV": False})

    csp_instance.add_constraint(A1D1Constraint("A1", "D1"))
    csp_instance.add_constraint(A1D2Constraint("A1", "D2"))
    csp_instance.add_constraint(A1D3Constraint("A1", "D3"))
    csp_instance.add_constraint(A2D1Constraint("A2", "D1"))
    csp_instance.add_constraint(A2D2Constraint("A2", "D2"))
    csp_instance.add_constraint(A2D3Constraint("A2", "D3"))
    csp_instance.add_constraint(A3D1Constraint("A3", "D1"))
    csp_instance.add_constraint(A3D2Constraint("A3", "D2"))
    csp_instance.add_constraint(A3D3Constraint("A3", "D3"))

    # symmetric constrains
    csp_instance.add_constraint(A1D1Constraint("D1", "A1"))
    csp_instance.add_constraint(A1D2Constraint("D1", "A2"))
    csp_instance.add_constraint(A1D3Constraint("D1", "A3"))
    csp_instance.add_constraint(A2D1Constraint("D2", "A1"))
    csp_instance.add_constraint(A2D2Constraint("D2", "A2"))
    csp_instance.add_constraint(A2D3Constraint("D2", "A3"))
    csp_instance.add_constraint(A3D1Constraint("D3", "A1"))
    csp_instance.add_constraint(A3D2Constraint("D3", "A2"))
    csp_instance.add_constraint(A3D3Constraint("D3", "A3"))

    solution = csp_instance.btsearch()
    if solution is None:
        print "No solution found!"
    else:
        print "\nSolution found:\n", solution

    # time evaluation
    elapsed = timeit.default_timer() - start_time
    print "\nExecution time:\t", round(elapsed, 2), "sec."


"""-------------------------------------------------------------------------"""
"""map coloring puzzle test:"""


def testColoringPuzzl():

    print "\ntesting map coloring:\n"
    start_time = timeit.default_timer()

    variables = ["Western Australia", "Northern Territory", "South Australia", "Queensland", "New South Wales", "Victoria",
                 "Tasmania"]
    domains = {}
    for variable in variables:
        domains[variable] = ["red", "green", "blue"]

    heuristic_flags = {"MRV": True, "LCV": True}
    csp_instance = csp.CSP(variables, domains, heuristic_flags)

    csp_instance.add_constraint(MapColoringConstraint("Western Australia", "Northern Territory"))
    csp_instance.add_constraint(MapColoringConstraint("Western Australia", "South Australia"))
    csp_instance.add_constraint(MapColoringConstraint("South Australia", "Northern Territory"))
    csp_instance.add_constraint(MapColoringConstraint("Queensland", "Northern Territory"))
    csp_instance.add_constraint(MapColoringConstraint("Queensland", "South Australia"))
    csp_instance.add_constraint(MapColoringConstraint("Queensland", "New South Wales"))
    csp_instance.add_constraint(MapColoringConstraint("New South Wales", "South Australia"))
    csp_instance.add_constraint(MapColoringConstraint("Victoria", "South Australia"))
    csp_instance.add_constraint(MapColoringConstraint("Victoria", "New South Wales"))
    csp_instance.add_constraint(MapColoringConstraint("Victoria", "Tasmania"))

    assignment = {}
    #solution = csp_instance.backtracking_search(assignment)
    solution = csp_instance.btsearch()

    if solution is None:
        print "No solution found!"
    else:
        print "\nSolution found:\n", solution

    # time evaluation
    elapsed = timeit.default_timer() - start_time
    print "\nExecution time:\t", elapsed, "sec."

"""-------------------------------------------------------------------------"""
"""The eight queens test:"""


def testEightQueens(n = 8):

    print "\ntesting Eight queens:\n"
    start_time = timeit.default_timer()

    # n = how many queens
    columns = list(xrange(n))
    rows = {}
    for column in columns:
        rows[column] = list(xrange(n))

    csp_instance = csp.CSP(columns, rows)

    csp_instance.add_constraint(EightQueensConstraint(columns))

    assignment = {}
    solution = csp_instance.btsearch()

    if solution is None:
        print "No solution found!"
    else:
        print "\nSolution found:\n", solution

    # time evaluation
    elapsed = timeit.default_timer() - start_time
    print "\nExecution time:\t", elapsed, "sec."

"""-------------------------------------------------------------------------"""
"""The n queens test - Minimum Conflicts implementation:"""
def testNQueens( n = 50):

    print "\ntesting Eight queens with Minimum Conflicts implementation:\n"
    start_time = timeit.default_timer()

    # n = how many queens
    columns = list(xrange(n))
    rows = {}
    for column in columns:
        rows[column] = list(xrange(n))

    csp_instance = csp.CSP(columns, rows)

    csp_instance.add_constraint(EightQueensConstraint(columns))

    # here we can see the different approach,
    # given initial (WRONG) assignment to be fixed by given maximum steps.
    assignment = {k: randrange(0, n) for k in range(n)}
    print "chosen random invalid initial assignment:\n",assignment ,"\n"
    maximum_steps_allowed = n * 10
    solution = csp_instance.MinConflicts(maximum_steps_allowed, assignment)

    if solution is None:
        print "No solution found!"
    else:
        print "\nSolution found:\n", solution

    # time evaluation
    elapsed = timeit.default_timer() - start_time
    print "\nExecution time:\t", elapsed, "sec."


testCrossword()
testColoringPuzzl()
testEightQueens()
testNQueens(25)
