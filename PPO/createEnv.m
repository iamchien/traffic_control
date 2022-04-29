function [env,obsInfo,actInfo] = createEnv()
    %% Step 1: create RL environment from Matlab template
    env = DrivingScenarioEnv;
    %% specifiy traffic problem formulation
    % specifiy action space
    env.TrafficSignalDesign = 1; % or 2, 3
    % the dimensio of signal phase is 3 for design 1 and 2, 4 for design 3, as
    % shown in the figure
    SignalPhaseDim = 3;
    env.phaseDuration = 50;
    env.clearingPhase = true;
    env.clearingPhaseTime = 5;
    % specifiy observation space
    env.ObservationSpaceDesign = 1; % or 2
    % specify reward parameter
    % The car's speed below the threshold will be treated as waiting at the
    % intersection
    slowSpeedThreshold = 3.5;
    % Add penalty for frequent/unnecessary signal phase switch
    penaltyForFreqSwitch = 1;
    % parameter for car collision
    env.hitPenalty = 20;
    env.safeDistance = 2.25;
    % reward for car pass the intersection
    env.rewardForPass = 10;
    % obtain observation and action info
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
end