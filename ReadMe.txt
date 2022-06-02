Readme File 
------------------
Paper title: "Towards Interpretable-AI Policies using Evolutionary Nonlinear Decision Trees for Discrete
Action Systems" 

This package contains scripts to train Blackbox AI agents and generate 
datasets corresponding to the control problems discuss in the paper and train NLDT using
Open Loop and Closed Loop training.

Environmetns taken from OpenAI gym:
1) Cartpole-v0
2) MountainCar-v0
3) LunarLander-v2

Custom made environment:
--> "CarFollowing" Environment.
This environment can be loaded using: "from rl_envs.custom_envs import CarFollowing"  
command at the preamble of the Python script.


BlackBox RL Training:
-- Training of CartPole-v0 and Lunarlander-v2 is done using PPO algorithm.
Following script trains the DNN agent using PPO algorithm: "ann_agents/PPO.py"
Following script generates data from the DNN agent which is trained using PPO algorithm: "ann_agents/test.py"
Following script can be used to visualize the simulation: "ann_agents/PPO_simulations.py"


-- Training of CarFollowing is done using DDQN algorithm
Following script trains the DNN agent using DDQN algorithm: "ann_agents/ann_agent_training.py"
Following script generates data from the DNN agent which is trained using DDQN algorithm: "ann_agents/ann_agent_data_creation.py"
Following script can be used to visualize the simulation: "ann_agents/simulate_ann_agent.py"

-- Training of MountainCar-v0 is done using SARSA with tile encoding.
Following script is used to train the black-box AI using SARSA: "mountain_car_rl/MountainCar/mountainCar.py"
Following script is used to generate data from the black-box AI which is trained using SARSA: "mountain_car_rl/MountainCar/generate_data.py"
Following script can be used to visualize the simulation: "mountain_car_rl/simulate_agent.py"


Datasets Information:
All datasets are stored in "datasets" folder.
Datasets used in this pape have following columns: x_1, x_2, x_3, ... x_d, q_1, q_2, ..., q_c, a
where "x_i" is the i-th state variable, "q_i" is the q-value or p-value output form the DNN for "i-th" action and "a" is the action value.
	- Training:
		CartPole-v0_100.data; CartPole-v0_500.data; CartPole-v0_1000.data; CartPole-v0_5000.data; CartPole-v0_10000.data;
		CarFollowing_10000.data
		MountainCar.data // regular data
		MountainCar_balanced.data // balanced data
		LunarLander-v2_10000.data // regular data
		LunarLander-v2_10000_balanced_data.data // balanced data
		
	- Testing
		CartPole-v0_testing_10000.data
		CarFollowing_testing_10000.data
		MountainCar_testing.data
		LunarLander-v2_testing_10000.data
		
		
The source codes for NLDT training are in the "main_codes" folder. Relevant  files are as follows:
- Open Loop Training:
	- main_serial_code.py (for running experiments on single core)
	- main_parallel_code.py (for running experiments with multiple runs on multiple cores)
- Closed Loop Training:
	- fine_tune_dt.py



Thanks.
Authors		
		
