clear; close all; clc;

figure();sgtitle("Traffic Control Using Reinforcement Learning Algorithms",'FontSize',20)

load 'DataOutput\DDPG_Ind.mat'
load 'DataOutput\DDPG_Reward.mat'
rewards = [agent_results(1,:), agent_results(2,:), agent_results(3,:), agent_results(4,:), agent_results(5,:)];
subplot(2,2,1); plot(idx, rewards); grid on;
title("DDPG")
xlabel("Steps"); ylabel("Rewards");

load 'DataOutput\DQN_Ind.mat'
load 'DataOutput\DQN_Reward.mat'
subplot(2,2,2); plot(idx, rewards); grid on
title("DQN")
xlabel("Steps"); ylabel("Rewards");

load 'DataOutput\SAC_Ind.mat'
load 'DataOutput\SAC_Reward.mat'
subplot(2,2,3); plot(idx, rewards); grid on
title("SAC")
xlabel("Steps"); ylabel("Rewards");

load 'DataOutput\SACD_Ind.mat'
load 'DataOutput\SACD_Reward.mat'
rewards = [agent_results(1,:), agent_results(2,:), agent_results(3,:), agent_results(4,:), agent_results(5,:)];
subplot(2,2,4); plot(idx, rewards); grid on
title("SAC-D")
xlabel("Steps"); ylabel("Rewards");
