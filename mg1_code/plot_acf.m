% Plot autocorrelation functions to decide cutoffs

clear all;

scen = 'rare';

shift = 1;
range = 1;
rate = 1;
numupd_met = 16;

run = strcat(scen,'_', num2str(shift),'_', num2str(range), '_', num2str(rate), '_', num2str(numupd_met));
acf_file = strcat('acf_', run);

load(acf_file);

tau_vec = zeros(1, 3);
use_lags = [300, 600, 300];
max_lags = 40000;

figure; plot(zeros(1, max_lags), 'black'); hold on;

plot(acf_mean_fft(2:max_lags, 1), 'blue'); hold on;
plot(acf_mean_fft(2:max_lags, 2), 'green');
plot(acf_mean_fft(2:max_lags, 3), 'red');

for i = 1 : 3

    tau_vec(i) = sum(acf_mean_fft(2:use_lags(i)+1, i)) * 2 + 1;

end

save(acf_file, 'tau_vec', 'use_lags', '-append');