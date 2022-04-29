function agent = createDDPG(obsInfo, actInfo)
    % Network structure inspired by original 2015 DDPG paper 
    % "Continuous Control with Deep Reinforcement Learning", Lillicrap et al.
    % https://arxiv.org/pdf/1509.02971.pdf

    %% CRITIC
    % Create the critic network layers
    actInfo.Dimension = [length(actInfo.Elements) 1];
    numObs = obsInfo.Dimension(1); numAct = actInfo.Dimension(1);
    statePath = [
        imageInputLayer([numObs 1],'Normalization','none','Name', 'observation')
        fullyConnectedLayer(256, 'Name', 'CriticStateFC1')
        reluLayer('Name','CriticStateRelu1')
        fullyConnectedLayer(256, 'Name', 'CriticStateFC2')
        ];

    actionPath = [
        imageInputLayer([numAct 1],'Normalization','none', 'Name', 'action')
        fullyConnectedLayer(256, 'Name', 'CriticActionFC1')
        ];
    
    commonPath = [
        additionLayer(2,'Name','add')
        reluLayer('Name','CriticCommonRelu1')
        fullyConnectedLayer(1, 'Name', 'CriticOutput')
        ];

    % Connect the layer graph
    criticNetwork = layerGraph(statePath);
    criticNetwork = addLayers(criticNetwork, actionPath);
    criticNetwork = addLayers(criticNetwork, commonPath);
    criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
    criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
    criticNetwork = dlnetwork(criticNetwork);

    % Create critic representation
    criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',5e-3, ... 
                                            'GradientThreshold',1,'L2RegularizationFactor',2e-4);
    
    criticOptions.UseDevice = 'cpu';
    critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
        'Observation','observation','Action','action');
                          
    %% ACTOR
    % Create the actor network layers
    actorNetwork = [
        imageInputLayer([numObs 1],'Normalization','none','Name','observation')
        fullyConnectedLayer(256, 'Name', 'ActorFC1')
        reluLayer('Name', 'ActorRelu1')
        fullyConnectedLayer(256, 'Name', 'ActorFC2')
        reluLayer('Name', 'ActorRelu2')
        fullyConnectedLayer(numAct, 'Name', 'ActorFC3')                       
        tanhLayer('Name','ActorTanh1')
        ];
%     actorNetwork = layerGraph(actorNetwork);
    actorNetwork = dlnetwork(actorNetwork);
    
    % Create actor representation
    actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',5e-4, ... 
                                            'GradientThreshold',1,'L2RegularizationFactor',2e-4);
    actorOptions.UseDevice = 'cpu';
    actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo);
    
    agentOptions = createDDPGOptions();
    agentOptions.ActorOptimizerOptions = actorOptions;
    agentOptions.CriticOptimizerOptions = criticOptions;
    agent = rlDDPGAgent(actor,critic,agentOptions);
end