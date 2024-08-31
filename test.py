import gymnasium
import numpy as np
from matplotlib import pyplot as plt

environment = gymnasium.make('FrozenLake-v1',map_name='8x8', is_slippery=True, render_mode='human')
environment.reset()
Q_table = np.loadtxt("Q.txt",dtype=float)

rewards = 0

episodes = 10

reward_vec=np.zeros(episodes)
reward_sum=np.zeros(episodes)

for i in range(episodes):
    state=environment.reset()
    finished = False
    stuck = False
    first_reset = True

    while (not finished):

        #print(state)
        
        if type(state) is tuple:
            action = np.argmax(Q_table[state[0],:])
            
        else:
            action = np.argmax(Q_table[state,:])
        
        
        new_state,reward,finished,stuck,_ = environment.step(action)
        state = new_state
    if reward:
        rewards +=1
        reward_vec[i] = 1

success_rate = rewards*100/episodes
print("the success rate was: %.2f %%"%success_rate )

for j in range(episodes):
    reward_sum[j] = np.sum(reward_vec[max(0,j-10):j+1])
plt.plot(reward_sum)
plt.xlabel('number of episodes')
plt.ylabel('total rewards as the episodes end')
plt.xlim([0,9])
plt.show()



