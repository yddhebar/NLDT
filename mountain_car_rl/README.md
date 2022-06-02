# OpenAIGym-Solutions
A collections of solutions for openai gym environments, using mainly reinforcement learning techniques.

## [MountainCar-v0](https://github.com/amohamed11/OpenAIGym-Solutions/tree/master/MountainCar)  
As of 12/10/2018 11:30PM MDT, currently #1 on the leaderboard  

![](https://i.postimg.cc/mgcc8Z6g/Mountain-Car-Leaderboard-Proof.png)  

### Agent:
A Sarsa(λ) Agent with eligibility trace and linear function approximation.  
Generating feature vectors was done with the [tiles3](http://incompleteideas.net/tiles/tiles3.html) library.

### Environment 
The bulk of the environment code is from the [RandomAgent](https://github.com/openai/gym/blob/master/examples/agents/random_agent.py) example on the OpenAI github.   
Modified to run the agent as necessary.  
Max Episode = 1000  

### Solution 
Required Avg score over 100 consecutive episodes = -110.00   
Achieved Avg score over 100 consecutive episodes = **-109.41**   
Number of Episodes for solution = **341**

[Video Link](https://streamable.com/sro3q)

![](https://github.com/amohamed11/OpenAIGym-Solutions/blob/master/MountainCar/rewardPerEpisode.png)


## References
* http://incompleteideas.net/tiles/tiles3.html
* https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
