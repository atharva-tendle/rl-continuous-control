def run_ddpg(n_episodes=2000, score_thres=30.0):
    """
    Runs Deep Q Learning.

    params:
        - n_episodes (int)    : max number of training episodes.
        - score_thres (float) : score required to solve the environment.
    """
    # list containing scores for each episode
    total_scores = []
    # last 100 scores
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        
        # setup lists for agents
        scores = np.zeros(num_agents)
        reacher_agent.reset()
        
        # reset environment 
        env_info = env.reset(train_mode=True)[brain_name]
        # get states for each agent
        states = env_info.vector_observations
        start = time.time()

        # utilize the agent
        while True:
            
            # get actions
            actions = reacher_agent.act(states)

            # take actions
            env_info = env.step(actions)[brain_name]

            # get observations for each agent
            next_states, rewards, dones, = env_info.vector_observations, env_info.rewards, env_info.local_done

            # update buffer and learn if necessary
            reacher_agent.step(states, actions, rewards, next_states, dones)

            # update state to new state
            states = next_states

            # increment reward
            scores += rewards

            # check if episode is done
            if np.any(dones):
                break
        
        #duration
        epoch_time = time.time() - start
        # get mean episode score
        ep_score = np.mean(scores)
        # save most recent mean score
        scores_window.append(ep_score)
        # save most recent mean score
        total_scores.append(ep_score)


        # print statements to keep track.
        print('\rEpisode {}\tAverage Score: {:.2f}\tTime taken: {}s'.format(i_episode, np.mean(scores_window), int(epoch_time)), end="")
        if i_episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}\tTime taken: {}s".format(i_episode, np.mean(scores_window), int(epoch_time)))
        
        # check if environment is solved
        if np.mean(scores_window) >= score_thres:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)))
            # save models for actor and critic
            torch.save(reacher_agent.actor_net.state_dict(), "actor_net.pth")
            torch.save(reacher_agent.critic_net.state_dict(), "critic_net.pth")
            break

    return scores