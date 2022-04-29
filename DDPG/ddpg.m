clear; close all; clc; %py.importlib.import_module('numpy');
folderName = fullfile(cd, 'OpenTrafficLab');
addpath(folderName);
%% Initialization
[env,obsInfo,actInfo] = createEnv();
action_dim = actInfo.Dimension(1);
action_dim = cast(action_dim,'uint8');
state_dim = obsInfo.Dimension(1);
state_dim = cast(state_dim,'uint8');
hidden_dim = 256;

replay_buffer_capacity = 5000;
replay_buffer = py.ddpg_2018.ReplayBuffer(replay_buffer_capacity);

%% Training Hyperparameters
RUNS = 5; EPISODE = 40; MAX_STEPS = 200;
agent_results = [];idx = [];
batch_size  = 256;
% figure; grid on; xlim([0 41000]); ylim([-500 500]);
% title("Traffic Control Using DDPG Algorithm"); xlabel('Steps'); ylabel('Rewards');

%% Training Loop
for i=1:RUNS
    ddpg_eval = py.ddpg_2018.DDPGAgent(state_dim, action_dim, replay_buffer, hidden_dim);
    rewards = [];
    for ep=1:EPISODE
        state = env.reset(); episode_reward = 0; done = false; step = 1;
        while step <= MAX_STEPS
            action = ddpg_eval.get_action(state);
            action = py2matlab(action);
            [next_state, reward, done, log] = env.step(action);
            ddpg_eval.replay_buffer.push(state, action, reward, next_state, done);

            if length(ddpg_eval.replay_buffer) > batch_size
                ddpg_eval.update_model(batch_size)
            else
                episode_reward = episode_reward + reward;
            end

            state = next_state;
            if done
                break
            end
            step = step + 1;
        end
        avg_reward = episode_reward / step;
        rewards(end+1) = avg_reward;
        if isempty(idx)
            idx(end+1) = step;
        else
            idx(end+1) = idx(end) + step;
        end
%         plot(idx, rewards)
    end
    agent_results = [agent_results; rewards];
end

% agent_results_py = py2matlab(agent_results);
% py.testMean.plotMean(agent_results_py,EPISODE,4)
