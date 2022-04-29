function phase = signalPhaseDesign2(action)
    % Copyright 2020 The MathWorks, Inc.
    % signal phase design 2: each phase has three lanes
    if action == 0
        phase = [1,1,1,0,0,1,...
                0,0,0,0,0,0]; % 9:20
    end
    if action == 1
        phase = [0,0,0,1,1,1,...
            0,0,1,0,0,0];
    end
    if action == 2
        phase = [0,0,0,0,0,0,...
            1,1,1,0,0,1];
    end
    if action == 3
        phase = [0,0,1,0,0,0,...
            0,0,0,1,1,1];
    end 
end