function [scenario] = createTJunctionScenario()
% Copyright 2020 The MathWorks, Inc.
% createDrivingScenario Returns the drivingScenario defined in the Designer

% Generated by MATLAB(R) 9.9 (R2020b) and Automated Driving Toolbox 3.2 (R2020b).
% Generated on: 16-Jun-2020 13:01:51

% Construct a drivingScenario object.
scenario = drivingScenario;
roadLength = [50 50 50];
% Add all road segments
roadCenters = [-401.4941 2185.809 0;
    -414.7725 2170.853 0];
roadDirection = diff(roadCenters)./norm(diff(roadCenters));
roadCenters(2,:) = roadCenters(1,:)+roadLength(1)*roadDirection;
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

roadCenters = [-402.2 2206.4 0;
    -417.7083 2220.263 0];
roadDirection = diff(roadCenters)./norm(diff(roadCenters));
roadCenters(2,:) = roadCenters(1,:)+roadLength(2)*roadDirection;
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

roadCenters = [-381.8 2208.1 0;
    -368.298 2223.199 0];
roadDirection = diff(roadCenters)./norm(diff(roadCenters));
roadCenters(2,:) = roadCenters(1,:)+roadLength(3)*roadDirection;
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

%rg = driving.scenario.RoadGroup('Name', 'T junction');
roadCenters = [-401.4941 2185.809 0;
    -381.5764 2208.243 0];
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

roadCenters = [-401.4941 2185.809 0;
    -400.5184 2186.908 0;
    -398.6231 2189.23 0;
    -399.4963 2203.927 0;
    -401.6534 2206.009 0;
    -402.7523 2206.984 0];
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

roadCenters = [-381.5764 2208.243 0;
    -382.5521 2207.144 0;
    -384.6341 2204.987 0;
    -399.3306 2204.114 0;
    -401.6534 2206.009 0;
    -402.7523 2206.984 0];
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

roadCenters = [-381.5764 2208.243 0;
    -401.4941 2185.809 0];
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

roadCenters = [-402.7523 2206.984 0;
    -401.6534 2206.009 0;
    -399.4963 2203.927 0;
    -398.6231 2189.23 0;
    -400.5184 2186.908 0;
    -401.4941 2185.809 0];
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);

roadCenters = [-402.7523 2206.984 0;
    -401.6534 2206.009 0;
    -399.3306 2204.114 0;
    -384.6341 2204.987 0;
    -382.5521 2207.144 0;
    -381.5764 2208.243 0];
marking = [laneMarking('Unmarked')
    laneMarking('Unmarked')
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Dashed', 'Width', 0.13)
    laneMarking('Solid', 'Width', 0.13)
    laneMarking('Unmarked')
    laneMarking('Unmarked')];
laneSpecification = lanespec([3 3], 'Width', [1.5 0.15 3.65 3.65 0.15 1.5], 'Marking', marking);
road(scenario, roadCenters, 'Lanes', laneSpecification);



        

