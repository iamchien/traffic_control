clear; close all; clc;

rng('default')
r = rand(1,1000);
len = length(r);
figure; grid on;
idx = [];

for i=1:len
    idx(end + 1) = i;
    if mod(i,100) == 0
        plot(idx, r(1:i));
        refreshdata
        drawnow
    end
end

    