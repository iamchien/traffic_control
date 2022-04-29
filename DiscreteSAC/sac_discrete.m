clear; close all; clc; %py.importlib.import_module('numpy');
folderName = fullfile(cd, 'OpenTrafficLab');
addpath(folderName);
%% Initialization
[env,obsInfo,actInfo] = createEnv();
action_dim = length(actInfo.Elements);
action_dim = cast(action_dim,'uint8');
state_dim = obsInfo.Dimension(1);
state_dim = cast(state_dim,'uint8');
hidden_dim = 256;

replay_buffer_capacity = 5000;
replay_buffer = py.sac_discrete.ReplayBuffer(replay_buffer_capacity);

%% Training Hyperparameters
TRAINING_EVALUATION_RATIO = 4;
RUNS = 5; EPISODES_PER_RUN = 400; STEPS_PER_EPISODE = 200;
idx = []; agent_results = [];

%% Training Loop
for run=1:RUNS
    agent = py.sac_discrete.SACAgent(state_dim, action_dim,replay_buffer);
    rewards = [];
    for episode_number=1:EPISODES_PER_RUN
        evaluation_episode = mod(episode_number , TRAINING_EVALUATION_RATIO) == 0;
        episode_reward = 0; state = env.reset(); done = false; i = 1;
        while i <= STEPS_PER_EPISODE
            action = agent.get_action(state, evaluation_episode);
            action = py2matlab(action.tolist());
            [next_state, reward, done, info] = env.step(action);
            if ~evaluation_episode
                agent.update_model(state, action, next_state, reward, done);
            else
                episode_reward = episode_reward + reward;
            end
            state = next_state;
            if done
                break
            end
            i = i + 1
        end
        if evaluation_episode
            rewards = [rewards,episode_reward];
            if isempty(idx)
                idx(end+1) = i;
            else
                idx(end+1) = idx(end) + i;
            end
        end
    end
    agent_results = [agent_results; rewards];
end

agent_results_py = matlab2py(agent_results);
py.testMean.plotMean(agent_results_py,EPISODES_PER_RUN,TRAINING_EVALUATION_RATIO)