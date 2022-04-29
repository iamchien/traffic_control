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

replay_buffer_capacity = 1000000;
replay_buffer = py.sac_modified_2019.ReplayBuffer(replay_buffer_capacity);

%% Training Hyperparameters
sac_eval = py.sac_modified_2019.SoftActorCritic('Deterministic',state_dim, action_dim, replay_buffer);
max_frames  = 40000;
max_steps   = 500;
frame_idx   = 0;
rewards     = []; idx = [];
batch_size  = 256;

figure; grid on; xlim([0 max_frames]); ylim([-500 500]);
%% Training Loop
while frame_idx < max_frames
    state = env.reset();
    episode_reward = 0;
    
    for step=1:max_steps
        action = sac_eval.get_action(state);
        action = py2matlab(action);
        [next_state, reward, done, log] = env.step(action);
        sac_eval.replay_buffer.push(state, action, reward, next_state, done);

        if length(sac_eval.replay_buffer) > batch_size
            sac_eval.soft_q_update(batch_size);
        end
        
        state = next_state;
        episode_reward = episode_reward + reward;
        sac_eval.save_checkpoint()
        frame_idx = frame_idx + 1
        
        if mod(frame_idx, 1000) == 0
            idx(end+1) = frame_idx; 
            rewards(end + 1) = episode_reward;
            plot(idx, rewards)
            refreshdata
            drawnow
        end
        
        if done
            break
        end
    end
end