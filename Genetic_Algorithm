import heapq, random

"""
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


def fitness(chrom):
    group_A = []
    group_B = []
    for index in range(0,10):
        if chrom[index]==0:
            group_A.append(index+1)
        else:
            group_B.append(index+1)

    sum=0
    for num in group_A:
        sum += num

    fit_fun1 = abs(36 - (sum))
    #print sum , "\t" , fit_fun1

    mul=1
    for num in group_B:
        mul *= num

    fit_fun2 = abs(360 - (mul))
   # print mul , "\t" , fit_fun2

    return fit_fun1 + fit_fun2

# function for implementing the single-point crossover
def crossover(l, q):
    # converting the string to list for performing the crossover
    l = list(l)
    q = list(q)

    # generating the random number to perform crossover
    k = random.randint(0, 9)

    # interchanging the genes
    for i in range(k, len(q)):
        l[i], q[i] = q[i], l[i]

    return l, q

def mutation(chrom):
    # generating the random number to perform crossover
    k1 = random.randint(0, 9)
    k2 = random.randint(0, 9)

    if k1%2==0 and k2%2==0:
        k3 = random.randint(0, 9)
    #    print "mutation accuerd in index:" , k3
        if chrom[k3] == 1:
            chrom[k3] = 0
        else:
            chrom[k3] = 1


def generation(chromosom_list):
    best_ch1 = heapq.heappop(chromosom_list)
    best_ch2 = heapq.heappop(chromosom_list)

   # print best_ch1 , "\t" ,best_ch2

    res1 , res2 = crossover(best_ch1[1], best_ch2[1])
   # print res1, "\t" , res2

    mutation(res1)
    mutation(res2)

    heapq.heappush(chromosom_list, (fitness(res1) , res1))
    heapq.heappush(chromosom_list, (fitness(res2) ,res2))

ch = [1,0,1,0,1,0,1,0,1,0]

chromosom_list = []

for i in range (50):
    # deep copy to new list
    new_ch = ch[:]
    # shuffle the new list
    random.shuffle(new_ch)
    ch_fitness = fitness(new_ch)
    heapq.heappush(chromosom_list, (ch_fitness,new_ch))


for i in range(5000):
    print "#",i," Genaration"
    if chromosom_list[0][0]==0:
        print "found in ", i , " Genaration."
        print "solution is: " , chromosom_list[0]
        break
  #  print chromosom_list
    generation(chromosom_list)

