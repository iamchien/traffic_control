function agentOpts = createDDPGOptions()
    % Copyright 2019 The MathWorks, Inc.

    %% DDPG Agent Options
    agentOpts = rlDDPGagentOpts;
    agentOpts.TargetUpdateFrequency = 1; 
    agentOpts.DiscountFactor = 0.99;
    agentOpts.MiniBatchSize = 128;
    agentOpts.ExperienceBufferLength = 1e6;
    agentOpts.TargetSmoothFactor = 1e-3;
    agentOpts.NoiseOptions.MeanAttractionConstant = 5;
    agentOpts.NoiseOptions.Variance = 0.5;
    agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
end