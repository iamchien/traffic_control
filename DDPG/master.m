% Copyright 2020 The MathWorks, Inc.
% Author: xiangxuezhao@gmail.com
% Last modified: 08-27-2020
% add traffic simulator: OpenTrafficLab to path
%% Step 1: create RL
% folderName = fullfile(cd, 'OpenTrafficLab');
% addpath(folderName)
clear; close all; clc;
% environment from Matlab template
[env,obsInfo,actInfo] = createEnv();
%% Step 3: creat DQN agent
agent = createDDPG(obsInfo, actInfo);
%% Step 4: train agent
% specify training option
trainOpts = createTrainOpts();
% train agent or load existing trained agent
doTraining = true;
if doTraining    
    % Train the agent.
    trainingInfo = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    folderName = cd; % change current folder
    folderName = fullfile(folderName, 'savedAgents');
    filename = strcat('TjunctionDQNAgentDesign', num2str(env.TrafficSignalDesign), '.mat');
    file = fullfile(folderName, filename);
    load(file);
end
%% Step 5: simulate agent
simOpts = rlSimulationOptions('MaxSteps',1000);
experience = sim(env,agent,simOpts);
totalReward = sum(experience.Reward);