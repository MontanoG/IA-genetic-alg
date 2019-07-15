from training import *
import controller
file_Name = 'data.pkl'
with open(file_Name,'rb') as f:
	pool, num, best_chrom, best_fitness = pickle.load(f)

print("FINAL POOL:")
print("Size of pool: {}".format(len(pool)))

# loop for all individuals
# for i in pool:
# 	print("\n",i.fitness)
# 	print(i.allele)

print("Current gen {}".format(num))
chrom = pool[0]
print("fitness of best chromosome {}".format(chrom.fitness))

print(len(best_chrom))
print([i.fitness for i in best_chrom])

T = 2
from osim.env import L2RunEnv as RunEnv
e = RunEnv(visualize=True)
# e = RunEnv(visualize=False)
e.reset()
total_reward = 0
total_reward_aux = 0
for t in range(700):
    obs, reward, done, _ = e.step(controller.input(chrom.allele,T,t*0.01))
    total_reward += reward
    if done:
        print("Done, {} steps".format(t))
        break
print(total_reward)

import matplotlib.pyplot as plt
# Best fitness
# print(best_fitness)
plt.plot(best_fitness)
plt.ylabel('Recompensa',fontsize='large')
plt.xlabel('Numero de generaciones',fontsize='large')
plt.title('Evolucion del mejor cromosoma')
plt.show()

print("END")
