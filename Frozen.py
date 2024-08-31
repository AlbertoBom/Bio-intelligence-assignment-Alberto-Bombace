import gymnasium
import numpy as np
from matplotlib import pyplot as plt



environment = gymnasium.make('FrozenLake-v1',map_name='8x8', is_slippery=True, render_mode=None)
environment.reset()

#Q table initialised
Q_table = np.zeros((64,4))

#Parameters
episodes = 30000
learn_rate = 0.5
discount = 0.9
rand = 1.0
decay= 0.00005

reward_vec = np.zeros(episodes)
reward_sum = np.zeros(episodes)

print("Q_table initialised")
print('\n')
print(Q_table)

#Train
for i in range(episodes):
    state=environment.reset()
    finished = False
    stuck = False
    first_reset = True

    while (not finished and not stuck):
        #explore
        rnd = np.random.random()

        if rnd<rand:
            action = environment.action_space.sample()
        else:
            if first_reset:
                action = np.argmax(Q_table[state[0],:]) 
            else:
                action = np.argmax(Q_table[state,:])
        
        
        new_state,reward,finished,stuck,_ = environment.step(action)
        #print(state)
        #print("i =%d" %i)
        #print(new_state)

        #Update Q table
        #every time the environment is reset the first state is a tuple and not just a number
        if first_reset:
            Q_table[state[0],action] = Q_table[state[0],action] + learn_rate*(reward + discount*np.max(Q_table[new_state,:]) - Q_table[state[0],action])
            state = new_state
            first_reset = False
        else:
            Q_table[state,action] = Q_table[state,action] + learn_rate*(reward + discount*np.max(Q_table[new_state,:]) - Q_table[state,action])
            state = new_state
    
    rand = max(rand-decay,0)
    
    if reward:
        reward_vec[i]=1




print("------------------------------------")
print('\n')
print(Q_table)

'''file=open("Q.txt",'w+')
file.write(str(Q_table))
file.close()'''
np.savetxt("Q.txt",Q_table)


for j in range(episodes):
    reward_sum[j] = np.sum(reward_vec[max(0,j-100):j+1])
plt.plot(reward_sum)
plt.xlabel('number of episodes')
plt.ylabel('reward per 100 episodes')
plt.show()

                             
