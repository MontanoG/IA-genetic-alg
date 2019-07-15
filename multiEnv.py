from multiprocessing import Pool
import controller
from numpy import exp as exp

def env(chrom):
    from osim.env import L2RunEnv as RunEnv
    e = RunEnv(visualize=False)
    e.reset()
    
    T = 2
    total_reward = 0
    for t in range(500):
        obs, reward, done, _ = e.step(controller.input(chrom.allele,T,t*0.01))
        total_reward += reward
        if done:
            break
    # print("HEADLESS: The reward is {}".format(total_reward))

    # enables to calculate accumulated fitness
    if total_reward < 0: total_reward = 0
    del e
    return total_reward
                

def PoolWorkers(num_core, pool_chrom):
    pool_workers = Pool(processes = num_core)

    return pool_workers.map(env,pool_chrom)
