clear; close all; clc; %py.importlib.import_module('numpy');
folderName = fullfile(cd, 'OpenTrafficLab');
addpath(folderName);
%% Initialization
[env,obsInfo,actInfo] = createEnv();
action_dim = length(actInfo.Elements);
% action_dim = matlab2py(cast(action_dim,'uint8'));
state_dim = obsInfo.Dimension(1);
% state_dim = matlab2py(cast(state_dim,'uint8'));
hidden_dim = 256;

buffer = py.utlis.RolloutBuffer();
max_steps = 500;

%% Training Hyperparameters
rewards_avg = []; idx = [];max_frames = 40000;frame_idx = 0; rewards = [];
print_running_reward = 0;print_running_episodes = 0;

%% Training Loop
ppo_agent = py.ppo.PPOAgent(state_dim,hidden_dim,action_dim,buffer);
figure; grid on; xlim([0 max_frames]); ylim([-500 500]);
while frame_idx < max_frames
    state = env.reset();current_ep_reward = 0;
    for t=1:max_steps
        action = ppo_agent.get_action(state);
        action = py2matlab(action);
        [state, reward, done, info] = env.step(action);

        ppo_agent.buffer.rewards.append(reward);
        ppo_agent.buffer.is_terminals.append(done);

        frame_idx = frame_idx + 1;
        current_ep_reward = current_ep_reward + reward;

        if mod(frame_idx,512) == 0
            ppo_agent.update_model();
        end
        if mod(frame_idx,512) == 0
            print_avg_reward = print_running_reward / print_running_episodes;
            print_avg_reward = round(print_avg_reward, 2);
            rewards(end + 1) = print_running_reward;
            idx(end+1) = frame_idx; 
            rewards_avg(end + 1) = print_avg_reward;
            plot(idx, rewards)
            refreshdata
            drawnow
            
            print_running_reward = 0;
            print_running_episodes = 0;
        end
        if done
            break
        end
    print_running_reward = print_running_reward + current_ep_reward;
    print_running_episodes = print_running_episodes + 1;
    end
end