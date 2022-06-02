import argparse
import sys
from mountain_car_rl.MountainCar.SarsaLambdaAgent import SarsaLambdaAgent
import gym
from gym import wrappers, logger
import numpy as np
import pickle


def video_schedule(episode_id):
    return episode_id % 50 == 0 or episode_id % 340 == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env_orig = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/SarsaControAgent-results'
    #env = wrappers.Monitor(env_orig, directory=outdir, force=True, video_callable=video_schedule)
    env = env_orig
    env.seed(0)
    np.random.seed(0)
    agent = SarsaLambdaAgent()

    episode_count = 2000
    episode_rewards = np.zeros(episode_count)
    reward = -1
    done = False
    i = 0
    np.ALLOW_THREADS = 2
    print('Learning Starts....')
    while (i < episode_count):
        if i >= 100:
            avg = np.average(episode_rewards[i-100:i])
            if avg > -110.0:
                break
        totalReward = 0
        observation = env.reset()
        while True:
            if totalReward == 0:
                # For step 1, only
                action = agent.start(observation)
                observation, reward, done, _ = env.step(action)
            else:
                action = agent.act(observation, reward, done)
                observation, reward, done, _ = env.step(action)

            totalReward += reward

            if done:
                agent.end(reward)
                episode_rewards[i] = totalReward
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        i += 1
        print('Episode = %d, Total Reward = %d' % (i + 1, totalReward))

    np.save("rewardPerEpisode", episode_rewards[:i])
    # Close the env and write monitor result info to disk

    #....Save the agent to disc..
    #pickle.dump(favorite_color, open("save.p", "wb"))
    pickle.dump(obj= agent, file=open('mountain_car_agent.p','wb'))
    env.close()