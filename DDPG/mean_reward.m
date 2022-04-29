clear all;
load 'DataOutput\DQN_Ind.mat'
load 'DataOutput\DQN_Reward.mat'

% load 'DataOutput\SACD_Reward.mat'
% load 'DataOutput\SACD_Ind.mat'
% agent_results = reshape(rewards,5, 40);
agent_results_py = matlab2py(agent_results);
EPISODE = 40;
setenv('TCL_LIBRARY', 'C:\Users\ngocm\AppData\Local\Programs\Python\Python37\tcl\tcl8.6')
setenv('TK_LIBRARY', 'C:\Users\ngocm\AppData\Local\Programs\Python\Python37\tcl\tk8.6')
py.testMean.plotMean(agent_results_py,EPISODE,4)
