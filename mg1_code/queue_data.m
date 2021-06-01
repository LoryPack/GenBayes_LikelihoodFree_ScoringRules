% Generate data from M/G/1 queue model

clear all;
s = RandStream('mt19937ar','Seed',0);
RandStream.setDefaultStream(s);

n = 2;

theta_1 = 1;
theta_2 = 2;
theta_3 = 0.01;

theta = [theta_1, theta_2, theta_3];

% Generate the interarrival times

w = exprnd(1/theta_3, 1, n);

% Generate the service times

u = theta_1 + (theta_2-theta_1)*rand(1, n);

% Compute the interdeparture times

y = zeros(1, n);
y(1) = w(1) + u(1);

for i = 2:n

    y(i) = u(i) + max(0, sum(w(1:i)) - sum(y(1:(i-1))));

end

save('queue_rare_short', 'y', 'w', 'u', 'theta');