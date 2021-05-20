import inspect
import operator
import random
import sys


class EvolutionProblem:
  """
  This class outlines the structure of a Evolutionery problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).

  given:
  - ps -> base population size.
  - gn -> number of generations.
  - cr -> rate of crossover.
  - rm -> rate of mutation.

  elements:
  - subject representation
  - base population as a set of subjects
  - subjects fitness

  """

  def __init__(self, initpop_size, initdna_size, genes_set, iterations, cross_rate=0.3, mute_rate=0.2, elitisem_rate=0.1, cool_after=500.0):
      """
      initialize a basic evolution problem.
      every detail in here is absolute to any G.A problem so we can generalize it.
      do note the specific implementations that we implemented in each problem, such as heuristics and fitness issues.
      """

      #key objects in a evolution problem
      self.dna = []
      self.DNA_LENGTH = initdna_size
      self.population = []
      self.INITIAL_POP_SIZE = initpop_size
      self.GENERATIONS_COUNT = iterations
      self.CUR_GENERATION = 0
      self.GENES = genes_set

      #creation of a random initial dna sequance
      for i in range(self.DNA_LENGTH):
          i = random.randint(0, self.GENES.__len__()-1)
          self.dna.append(self.GENES[i])

      #key factors
      self.MUTATION_RATE = mute_rate
      self.CROSSOVER_RATE = cross_rate
      self.CROSSOVER_COOL_FACTOR = float(self.GENERATIONS_COUNT) / cool_after  # lowering till the coolafter generation parameter and from than its close to 0 always
      self.ELITISEM_AMOUNT = int(round(elitisem_rate * self.INITIAL_POP_SIZE))
      self.STALEMATE_ALLOWED = 500
      self.STALEMATE_COUNT = 0
      self.CUR_FITTEST = 0
      self.PREV_FITTEST = 0
      self.SOLVED = False

      #stats
      self.STATISTICS_MAX_FITNESSES = []
      self.STATISTICS_MIN_FITNESSES = []
      self.STATISTICS_AVG_FITNESSES = []

  def generateInitialPopulation(self):
     """
     must implement individualy
     """
     self.raiseNotDefined()
  def sortPopulation(self):
      """

      :return:
      """
      self.population.sort(key=lambda x: x[0], reverse=True)
  def checkStalmation(self):

      # stop if we're in stalemation
      self.CUR_FITTEST = self.getFittest()[0]
      if self.PREV_FITTEST == self.CUR_FITTEST:
          self.STALEMATE_COUNT += 1
      else:
          self.STALEMATE_COUNT = 0

      if self.STALEMATE_COUNT >= self.STALEMATE_ALLOWED:
          print("Stopping after {} generations without any improvement (stalemate).".format(self.STALEMATE_COUNT))
          return True

      self.PREV_FITTEST = self.CUR_FITTEST
      if self.PREV_FITTEST != 0:
          best_fitness_inverted = 1.0 / self.PREV_FITTEST
      else:
          best_fitness_inverted = float('inf')
      #print("Gen {}...\tBest: '{}'\tFitness: (1/{:.2f})\tSolved: {}".format(self.CUR_GENERATION + 1, self.population[0][0], best_fitness_inverted, self.SOLVED))

      # updating generation
      self.CUR_GENERATION += 1

      return False
  def calculateStatistics(self):
      self.STATISTICS_MAX_FITNESSES.append(self.population[0][0])
      self.STATISTICS_MIN_FITNESSES.append(self.population[-1][0])
      self.STATISTICS_AVG_FITNESSES.append(float(sum([x[0] for x in self.population])) / self.INITIAL_POP_SIZE)
  def selectParent(self, method):

      # list of selected dna's - implementaion MUST return a selected list sorted as [[father],[mother],[father],[mother]...]
      selected = method

      return selected
  def defaultSelection(self):

      # nothing special, evryone can get selected.
      result = self.population[random.randint(0, self.population.__len__())]

      return result
  def tournamentSelection(self, k=0):

      k = self.INITIAL_POP_SIZE - 1
      index1 = random.randint(0, k)
      index2 = random.randint(0, k)

      while index2 == index1:
          index2 = random.randint(0, k)
      candidate1 = self.population[index1]
      candidate2 = self.population[index2]
      if candidate1[0] >= candidate2[0]:
          return candidate1
      else:
          return candidate2
  def fitnessProportionateSelection(self):

      max = sum(chromosome[0] for chromosome in self.population)
      pick = random.uniform(0, max)
      current = 0
      for chromosome in self.population:
          current += chromosome[0]
          if current > pick:
              return chromosome

      return chromosome
  def stochasticUniversalSamplingSelection(self, n=1):

      n = self.population.__len__()

      newlist = []

      for subject in self.population:
          object = [0, []]
          object[0] = subject[0] * -1
          object[1] = subject[1]
          newlist.append(object)

      new_population = sorted(newlist, key=operator.itemgetter(1))

      # print "\n\n", population, "\n\n"

      F = sum(i[0] for i in new_population)  # total fitness of population
      P = F / n  # distance between the pointers
      start = random.uniform(0.0, P)
      Points = [start + i * P for i in range(n)]

      keep = []
      for P in Points:
          i = 0
          while sum(new_population[j][0] for j in range(i + 1)) < P:
              i += 1
          keep.append(new_population[i])

      # print  self.groupOfSubjects.heap

      return keep
  def doMutation(self, dna):

      # generating the possibilty to perform mutation according to mutation rate
      for gene in range(dna.__len__()):
          if random.randint(0, 100) < self.MUTATION_RATE * 100:
              # what to put in it
              gene_index = random.randint(0, self.GENES.__len__()-1)
              while self.GENES[gene_index] == dna[gene]:
                  gene_index = random.randint(0, self.GENES.__len__()-1)
              dna[gene] = self.GENES[gene_index]

      return dna
  def doCrossover(self, subjectFather, subjectMother):
      """
      function for implementing the single-point crossover
      :param subjectFather:
      :param subjectMother:
      :return:
      """

      if random.randint(0, 100) < self.CROSSOVER_RATE * 100:
          # generating the random position to perform crossover
          pos = random.randint(0, len(subjectMother))

          # interchanging the genes
          crossed_dna1 = subjectFather[:pos] + subjectMother[pos:]
          crossed_dna2 = subjectMother[:pos] + subjectFather[pos:]

          return crossed_dna1, crossed_dna2
      else:
          return subjectFather, subjectMother
  def doCrossover_Kpoints(self, dna1, dna2, k):
      for i in range(k):
          dna1, dna2 = self.doCrossover(dna1, dna2)

      return dna1, dna2
  def doCrossover_uniform(self, dna1, dna2, p=0.5):
      "Uniform crossover"

      result1 = ""
      result2 = ""
      for i in range(self.DNA_LENGTH):
          if random.random() < p:
              result1 += dna1[i]
              result2 += dna2[i]
          else:
              result1 += dna2[i]
              result2 += dna1[i]

      return result1, result2
  def crossoverCooling(self):

      cooling_crossover_rate = 1 - (
                  (float(self.CUR_GENERATION) + 1) / (self.CROSSOVER_COOL_FACTOR * self.GENERATIONS_COUNT))

      if cooling_crossover_rate > 0:
          self.CROSSOVER_RATE = self.CROSSOVER_RATE * cooling_crossover_rate
  def getFitness(self, subject):
     """
     must implement individually
     """
     self.raiseNotDefined()
  def getFittest(self):

      top = self.population[0]

      return top
  def raiseNotDefined(self):
      """
      basic notification for an undefined method.
      """
      print "Method not implemented: %s" % inspect.stack()[1][3]
      sys.exit(1)


#servivalOfTheFittest
def geneticAlgorithm(EvolutionProblem, selectionMethod):
    """
    the algorithm itself, implemented abstractly and hopefully very clearly!
    """

    #GENERATE POPULATION
    EvolutionProblem.generateInitialPopulation(True)

    #CREATE NEW GENERATION
    for generation in range(EvolutionProblem.GENERATIONS_COUNT):

        #PROBLEM SOLVED
        if EvolutionProblem.isGoal():
            print "=>\tTermination:\tFound solution in #", generation, " Genaration."
            break

        #STALMATION
        if EvolutionProblem.checkStalmation():
            break

        #CREATE OFFSPRINGS
        offsprings = []
        while offsprings.__len__()<EvolutionProblem.INITIAL_POP_SIZE:

                #SELECTIOM
                parent1 = EvolutionProblem.selectParent(eval(selectionMethod))
                parent2 = EvolutionProblem.selectParent(eval(selectionMethod))

                #CROSSOVER
                offspring1, offspring2 = EvolutionProblem.doCrossover_Kpoints( parent1[1],  parent2[1],5)

                #MUTATION
                EvolutionProblem.doMutation(offspring1)
                EvolutionProblem.doMutation(offspring2)

                #COLLECT OFFSPRINGS
                offsprings.append([EvolutionProblem.getFitness(offspring1), offspring1])
                offsprings.append([EvolutionProblem.getFitness(offspring2), offspring2])


        #ELITISEM
        EvolutionProblem.population = EvolutionProblem.population[:EvolutionProblem.ELITISEM_AMOUNT]

        # ADD OFFSPRINGS
        offsprings.sort(key=lambda x: x[0], reverse=True)
        EvolutionProblem.population +=  offsprings

        #TRIMMING POPULATION TO INITIAL SIZE, CUTTING THE LEAST FIT
        EvolutionProblem.population = EvolutionProblem.population[:EvolutionProblem.INITIAL_POP_SIZE]
        EvolutionProblem.sortPopulation()

        #CALCULATE STATISTICS - OPTIONAL
        EvolutionProblem.calculateStatistics()

        #REDUCE AMOUNT OF CROSSOVER GIVEN TIME
        EvolutionProblem.crossoverCooling()
