import numpy as np
import pickle, copy
from bisect import bisect
import multiEnv

# constants
file_Name = 'data.pkl'
# number of cores
num_core = 8
# crossover alpha
alpha = 0.3
# mutation probability
pm = 0.5
# mutation max variation	
delta = 0.2
# population
pop_size = 200
# number of childs
num_ch = pop_size/2
# number of generations
gen = 500 

class chromosome():
	def __init__(self, allele):
		self.allele = copy.deepcopy(allele)
		self.fitness = None
		
	def mutation(self):
		for i in range(len(self.allele)):
			index = np.random.randint(len(self.allele[i]))
			self.allele[i][index]+=np.random.randn()*delta
		
def selection(pool):
	# total fitness 
	total_rwd = sum([i.fitness for i in pool])

	# random numbers for selecting parents
	thres = np.random.uniform(0.0, total_rwd, int(num_ch))
	
	prev = pool[0].fitness
	accum_fit = []
	accum_fit.append(prev)

	for chr in pool[1:]:
		accum_fit.append(prev + chr.fitness)
		prev = accum_fit[-1]

	# https://docs.python.org/3/library/bisect.html
	# selected parents
	selected = []
	for i in thres:
		j = bisect(accum_fit,i)
		if j != len(accum_fit): selected.append(pool[j])
	return selected
	
# cruzamiento completo
def crossover(ind1,ind2):
	a1 = copy.deepcopy(ind1.allele)
	a2 = copy.deepcopy(ind2.allele)

	child1 = [i*alpha+j*(1-alpha) for i,j in zip(a1,a2)]
	child2 = [i*alpha+j*(1-alpha) for i,j in zip(a2,a1)]
	
	return chromosome(child1), chromosome(child2)

def evaluation(subpop):
	rewards = multiEnv.PoolWorkers(num_core, subpop)
	for i in range(len(subpop)):
		subpop[i].fitness = rewards[i]

			
def generate_pool():
	return [chromosome([(np.random.randn(8)) for i in range(9)]) for j in range(pop_size)]

if __name__ == '__main__':
	try:
		with open(file_Name,'rb') as f:
			pool, num, best_chrom, best_fitness = pickle.load(f)
			gen -= num
		print("Best pool loaded!")
	except:
		pool = generate_pool()
		print("Evaluating new pool")
		evaluation(pool)
		best_fitness = []
		best_chrom = []
	evaluation(pool)

	for num in range(gen):
		print("\nGen. {}/{}".format(num,gen))
		# selection step
		selected = selection(pool)

		# crossover step
		childs = []
		for sp1,sp2 in zip(selected[0::2],selected[1::2]):
			childs.extend(crossover(sp1,sp2))
		
		# adding childs to mutation schedule
		# to_mutate = []
		# to_mutate.extend(childs)
		
		# random selection from pool to mutate
		prob = np.absolute(np.random.randn(len(pool)))

		to_mutate = [chromosome(copy.deepcopy(pool[i].allele)) 
		for i in range(len(pool)) if prob[i]<pm]

		to_mutate.extend(childs)

		# mutation step
		for ind in to_mutate:
			ind.mutation()
		
		# eval childs and mutants
		evaluation(to_mutate)
		pool.extend(to_mutate)
		
		# next generation
		pool.sort(key = lambda chromosome: chromosome.fitness, reverse = True)
		pool = pool[:pop_size]
		if num % 20==0: best_chrom.append(pool[0])
		best_fitness.append(pool[0].fitness)
			
		with open(file_Name,'wb') as f:
			pickle.dump([pool, num+1, best_chrom, best_fitness], f)
