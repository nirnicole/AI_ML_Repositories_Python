import Problem
import random


class CardsProblem(Problem.EvolutionProblem):
    """
        in this problem we get 10 cards, from 1 to 10.
        we need to sort them into 2 groups that forming a group that by multipcation gives 360, while the other is been added to sum to 36.
        ofcourse, no repitetions or absents is allowed.

        chromosom structure:
                  list of '0'/'1' representing if a card is in group A or B, the index is the card value between 1-10.

        fitness function:
                  f1(ch)= |36 - (g0+g1+g2+g3+g4)|
                  f2(ch)= |360 - (g5*g6*g7*g8*g9) |
                  f(ch) = |f1| + |f2|

        maximization problem:
                  f(ch) = |f1| + |f2| = 0 (or close as possible)

        arrtibutes:
            initial population = 50
            generations = 500
            Swap function = Crossing at one random point
            mutation probability = 0.25

        solution:    {1011110000} -> group A:{2,7,8,9,10} , group B:{1,3,4,5,6}
    """

    def __init__(self, initpop_size, iterations, cross_rate = 0.3, mute_rate = 0.2, elitisem_rate=0.1, cool_after=500.0):
        """
        """
        Problem.EvolutionProblem.__init__(self, initpop_size, 10 , [0,1], iterations, cross_rate, mute_rate, elitisem_rate, cool_after)

    def generateInitialPopulation(self, use_heuristics=False):
        """

        :return:
        """
        for i in range(self.INITIAL_POP_SIZE):
            # deep copy to new list
            new_dna = self.dna[:]
            # shuffle the new list
            random.shuffle(new_dna)
            #add to group with a default starting fitness evaluation(a bad one)
            self.population.append([self.getFitness(new_dna), new_dna])

    def getFitness(self, dna):

        group_A = []
        group_B = []
        for index in range(0, 10):
            if dna[index] == 0:
                group_A.append(index + 1)
            else:
                group_B.append(index + 1)

        sum = 0
        for num in group_A:
            sum += num

        fit_fun1 = abs(36 - (sum))

        mul = 1
        for num in group_B:
            mul *= num

        fit_fun2 = abs(360 - (mul))

        score = fit_fun1 + fit_fun2

        if score==0:
            self.SOLVED=True
            #self.paintSolution(dna)


        score = 1.0/(score+1)

        return score

    def isGoal(self):
        return self.SOLVED

#FUNCTIONS

    def paintSolution(self, dna):
          g1= []
          g2 =[]
          for index in range(len(dna)):
            if dna[index]==0:
              g1.append(index+1)
            else:
              g2.append(index+1)

          print g1
          print g2
