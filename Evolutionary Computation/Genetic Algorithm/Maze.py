import Problem
import random
from PIL import Image
import numpy as np
import math


class MazeProblem(Problem.EvolutionProblem):
    """
        in this problem we get different attempts to find a solution to a distance problem and we try to utilize them.
        its actually a local searching problem.

        dna structure:
                  list of chars representing the direction the robot went from the set: {U,D,L,R}.

        fitness function:
                  will be elaborate later.

        maximization problem:
                the fitness will be normalized between 0 to 1.

        attributes:
            initial population
            generations
            Swap function = Crossing at k random point
            mutation probability
            crossing probability
            elitism probability

    """

    def __init__(self,path, initpop_size, iterations,maze_diffic = 0, cross_rate = 0.3, mute_rate = 0.2, elitisem_rate=0.1,  cool_after=500.0, start=[0,0], end=[0,0]):
        """
        we need to add to the basic initialization the following factors,
        including a layout data to compare to, start point and ending.
        """
        self.layout_data= []
        self.layout_data = self.translateMaze(path)
        self.LAYOUT_LINES = self.layout_data.__len__()
        self.LAYOUT_COLUMNS = self.layout_data[0].__len__()
        self.MAX_DIMENSION = max(self.LAYOUT_LINES, self.LAYOUT_COLUMNS)
        self.MAZE_DIFFICULTY = maze_diffic

        Problem.EvolutionProblem.__init__(self, initpop_size, (maze_diffic+1)* self.MAX_DIMENSION , ['L', 'R', 'U', 'D'], iterations, cross_rate, mute_rate, elitisem_rate, cool_after)

        self.SOURCE_COORDINATE = start
        self.DESTINATION_COORDINATE = end
        self.layout_data[end[0]][end[1]]=2

    def generateInitialPopulation(self, use_heuristics=False):
        """
        if heuristics are permitted,
        half of the population will be biased and quit accurate to the destination.
        although if obstacles are around, they will go through them and loose a lot of credit.

        the rest of the population is randomized.
        """
        self.CUR_GENERATION = 0
        cutting_factor = 2

        #create with heuristics
        if use_heuristics:
            cutting_factor*=2

            #start with manhattan routes
            x1, y1 = self.SOURCE_COORDINATE
            x2, y2 = self.DESTINATION_COORDINATE

            steps_vertically = x2 - x1
            steps_horizontally = y2 - y1
            self.GENES = ['L', 'R', 'U', 'D']

            if steps_vertically>=0:
                vertical_direction = self.GENES[3]        #down
            else:
                steps_vertically *= -1
                vertical_direction = self.GENES[2]        #up

            if steps_horizontally>=0:
                horizontal_direction = self.GENES[1]        #right
            else:
                steps_horizontally *= -1
                horizontal_direction = self.GENES[0]        #left

            for i in range(2*self.INITIAL_POP_SIZE / cutting_factor):
                new_dna = []

                #creat manhattan route
                for _ in range(steps_vertically):
                    new_dna.append(vertical_direction)

                for _ in range(steps_horizontally):
                    new_dna.append(horizontal_direction)

                while new_dna.__len__()<self.DNA_LENGTH:
                    new_dna.append(self.GENES[3])

                # add to group with a starting fitness evaluation
                self.population.append([self.getFitness(new_dna), new_dna])

        #create random dna's
        for i in range(self.INITIAL_POP_SIZE/cutting_factor):
            action = i%4
            new_dna = [self.GENES[action] for gene in range(self.DNA_LENGTH)]
            #add to group with a starting fitness
            self.population.append([self.getFitness(new_dna), new_dna])
        while(self.population.__len__()<self.INITIAL_POP_SIZE):
            # deep copy to new list
            new_dna = self.dna[:]
            # shuffle the new list
            random.shuffle(new_dna)
            #add to group with a starting fitness
            self.population.append([self.getFitness(new_dna), new_dna])

    def getFitness(self, dna):
        """
        calculate how accurate the rout was,
        punish for obstacles ignorance and bonus every improving step.
        nonetheless, punish greatly for borders crossing.
        """

        REPEATED_PENALTY = 2  # each repeated node is considered the same as REPEATED_PENALTY steps
        OBSTACLE_PENALTY = 200  # each obstacle on the way is considered the same as OBSTACLE_PENALTY steps
        DEST_NOT_REACHED_PENALTY = 2*self.DNA_LENGTH + self.manhattanDistance(self.SOURCE_COORDINATE, self.DESTINATION_COORDINATE)  # if destination is not reached

        #collect data of route
        distance, bonus, repeated_nodes, obstacles, final_point, is_out_of_bound = self.simulateRoute(dna)

        # Calculate Penalties
        penalties = repeated_nodes * REPEATED_PENALTY + obstacles * OBSTACLE_PENALTY

        #if out of bound
        if is_out_of_bound:
            aggregate_fitness = float('inf')
        else:
            if distance == float('inf'):                  #if the rout didnt achive destination its weight will be its final point dis +penalties
                penalties += DEST_NOT_REACHED_PENALTY
                aggregate_fitness = self.manhattanDistance(final_point, self.DESTINATION_COORDINATE) + penalties
            else:
                aggregate_fitness = distance + penalties
            aggregate_fitness -= bonus


        #normalization
        normelized_fitness = 1.0 / (1+aggregate_fitness)

        if distance < float('inf') and not obstacles:
            print dna , (normelized_fitness)
            self.paintRout(dna)
            self.SOLVED = True

        # returning the inverse of the cost, so it'll become a proper fitness function
        return normelized_fitness

    def isGoal(self):

        if self.SOLVED :   #if its fitness is perfect
            return self.SOLVED
        return self.SOLVED

#FUNCTIONS

    def translateMaze(self, path, resize_factor=100.0):

        # 'maze.png' = path
        # Open the maze image and make greyscale, and get its dimensions
        try:
            im = Image.open(path).convert('L')
            w, h = im.size

            # Ensure all black pixels are 0 and all white pixels are 1
            binary = im.point(lambda p: p > 128 and 1)

            # Resize to about resize_factorXresize_factor pixels.
            resizing_w = int(math.ceil(float(w)/resize_factor))
            resizing_h = int(math.ceil(float(h)/resize_factor))
            binary = binary.resize((w // resizing_w, h // resizing_h), Image.NEAREST)
            w, h = binary.size

            # Convert to Numpy array - because that's how images are best stored and processed in Python
            nim = np.array(binary)

            mat = []
            # Print that puppy out
            for r in range(h):
                line = []
                for c in range(w):
                    line.append(nim[r, c])
                mat.append(line)

            count=0
            """
            print "\n\n"
            for i in range(mat.__len__()):
                count+=1
                print mat[i]
            print "\n"
            """
            return mat
        except:
            print("Couldn't find path in images directory.")
            print("For your Attatntion, this is the path to put your layouts:")
            print path
            print "try putting your layout there."

    def paintRout(self, gene):

        layout_copy = [row[:] for row in self.layout_data]

        line = self.SOURCE_COORDINATE[0]
        column = self.SOURCE_COORDINATE[1]

        layout_copy[line][column] = 6


        for action in gene:
            # goal, do step and stop counting
            if (line==self.DESTINATION_COORDINATE[0]) and (column==self.DESTINATION_COORDINATE[1]):
                layout_copy[line][column] = 8
                break

            #do the action {'L','R','U','D'}
            if action== 'L':
                column-=1
            elif action== 'R':
                column+=1
            elif action== 'U':
                line-=1
            elif action== 'D':
                line+=1

            #out of bounds
            if line<0:
                line+=1
            elif line>= self.LAYOUT_LINES:
                line-=1
            if column<0:
                column+=1
            elif column>= self.LAYOUT_COLUMNS:
                column-=1

            layout_copy[line][column] = 7

        layout_copy[line][column] = 8

        print "\n\nRout Visualization :\n"
        for i in range(layout_copy.__len__()):
            print layout_copy[i]
        print "\nend"

        return layout_copy

    def simulateRoute(self, route):
        """
        Returns a tuple of (distance, repeated_nodes, obstacles, final_point, is_out_of_bound).

        Distance will be infinity if the route is invalid (are taking us out of scope, or not reaching
        the destination)
        """
        source = self.SOURCE_COORDINATE
        destination = self.DESTINATION_COORDINATE
        NOT_REACHED = float('inf')

        if source == destination:
            return 0, 0, 0, source, False

        point = source
        steps = 0
        bonus = 0
        revisit = 0
        obstacles = 0
        initial_distance = self.manhattanDistance(source,destination)
        points_visited = [point]

        for action in route:
            steps += 1

            new_distance = self.manhattanDistance(point,destination)
            if new_distance < initial_distance:
                bonus +=1
                initial_distance = new_distance
            else:
                bonus-=1

            if action == "L":
                point = [point[0], point[1]-1]
            elif action == "R":
                point = [point[0], point[1] + 1]
            elif action == "U":
                point = [point[0]-1, point[1]]
            elif action == "D":
                point = [point[0]+1, point[1]]
            else:
                break

            #breaking out situations
            if point == destination:
                return steps ,3*bonus , revisit, obstacles, point, False

            if point[0] < 0 or point[0] >= self.LAYOUT_LINES or point[1] < 0 or point[1] >= self.LAYOUT_COLUMNS:
                # out of bounds, illegal move
                return NOT_REACHED ,bonus , revisit, obstacles, point, True

            #count obstacles
            if self.layout_data[point[0]][point[1]]==0:
                obstacles += 1

            #count visited
            if point in points_visited:
                revisit += 1
            else:
                points_visited.append(point)



        return NOT_REACHED ,bonus , revisit, obstacles, point, False

    def manhattanDistance(self, xy1, xy2):
        "Returns the Manhattan distance between points xy1 and xy2"
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
