import Cards, Maze, Problem
from sys import path

if __name__ == '__main__':
  """
  The main function called when chromTest.py is run
  from the command line:
  > python chromTest.py
  
  options for selection methods:
  1. defaultSelection - the simplest and fastest one, works best WITH NO CROSSOVERS AND 100% MUTATION RATE
  2. fitnessProportionateSelection - works best with 50% crossover rate and 100% mutation rate.
  3. stochasticUniversalSamplingSelection - worst with a big population, best with smallest of population,say 8,  and 100%  cross and mutate rate!
  4. tournamentSelection - works best with 100% crossover rate and 100% mutation rate.
  """
  selection_method = 'tournamentSelection'     #only code you should change!
  selection_method = 'EvolutionProblem.' + selection_method + '()'

  print "\nRunning Maze test:"
  print "Maze layouts should be saved on 'images' directory.\nyou can add layouts you created with paint to the directory and choose it by name.\nthe recommanded default is 'smallMaze.png'.\n"
  print "Please choose a maze from the \"images\" directory:"
  image_name = raw_input()
  image_path = path[0] + "\images\\" + image_name


  for i in range(1):
    problem = Maze.MazeProblem(image_path, 100 , 5000 ,1, 0.9, 0.1, 0.1, 1000, [2,26],[30,20])
    obj = Problem.geneticAlgorithm(problem,selection_method)

    TEST_NAME = "size{}pop{}maze{}".format(problem.MAX_DIMENSION, problem.INITIAL_POP_SIZE, image_name)

    with open('{}_max.txt'.format(TEST_NAME), 'w') as f:
        for n in problem.STATISTICS_MAX_FITNESSES: f.write("{}\n".format(n))
    with open('{}_min.txt'.format(TEST_NAME), 'w') as f:
        for n in problem.STATISTICS_MIN_FITNESSES: f.write("{}\n".format(n))
    with open('{}_avg.txt'.format(TEST_NAME), 'w') as f:
        for n in problem.STATISTICS_MIN_FITNESSES: f.write("{}\n".format(n))


  """
  print "\nRunning cards test:\n"
  iter = 100
  for i in range(100):
    problem = Cards.CardsProblem(20,100,0.9,0.1,0.1,500.0)  #(pop_size, numOf_iteratrions, crossover_rate,mutation_rate)
    obj = Problem.geneticAlgorithm(problem,selection_method)
  """

  pass
