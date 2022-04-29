clear; close all; clc; % py.importlib.import_module('dqn');
folderName = fullfile(cd, 'OpenTrafficLab');
addpath(folderName);
%% Initialization
[env,obsInfo,actInfo] = createEnv();
action_dim = length(actInfo.Elements);
action_dim = cast(action_dim,'uint8');
state_dim = obsInfo.Dimension(1);
state_dim = cast(state_dim,'uint8');
hidden_dim = 128;

layer_sizes = [state_dim, hidden_dim, action_dim];
layer_sizes = matlab2py(layer_sizes);
%% Training Hyperparameters
exp_replay_size = 256;
agent = py.dqn.DQN_Agent(1423, layer_sizes,1e-3,5,exp_replay_size);

% Main training loop
losses_list = []; reward_list = []; episode_len_list = [];
epsilon_list = []; episodes = 10000; epsilon = 1;

% initiliaze experiance replay
index = 0;
for i=1:exp_replay_size
    obs = env.reset();
    done = false;
    while ~done
        A = agent.get_action(obs, action_dim, 1);
        A = py2matlab(A)
        [obs_next, reward, done, log] = env.step(A);
        agent.collect_experience([obs, A, reward, obs_next]);
        obs = obs_next;
        index = index + 1;
        if index > exp_replay_size
            break
		end
	end
end

% Training Loop
index = 128; i = 1;
while i <= episodes
	obs = env.reset(); done = false; losses = 0; ep_len = 0; rew = 0;
	while ~done
        ep_len = ep_len + 1;
        A = agent.get_action(obs, action_dim, epsilon);
        A = py2matlab(A)
        [obs_next, reward, done, log] = env.step(A);
        agent.collect_experience([obs, A, reward, obs_next]);

        obs = obs_next; rew = rew + reward; index = index + 1;

        if index > 128
            index = 0;
            for j=1:4
                loss = agent.train(128);
                losses = losses + loss;
            end
        end
    end

    if epsilon > 0.05
        epsilon = epsilon - (1 / 5000);
    end

    losses_list = [losses_list, losses / ep_len]; reward_list = [reward_list, rew];
    episode_len_list = [episode_len_list, ep_len]; epsilon_list = [epsilon_list, epsilon];
	i = i + 1
end