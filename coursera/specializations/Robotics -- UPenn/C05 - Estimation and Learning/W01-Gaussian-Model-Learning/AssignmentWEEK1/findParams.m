function [mu, sigma, weights] = findParams

mu = zeros(1, 3, 'double');
sigma = zeros(3, 3, 'double');
weights = (1.0 / 3.0) + zeros(1, 3, 'double');

% Load Samples
loaded = load("train_samples.mat");
samples = loaded.Samples;
N = size(samples, 1);

% Apply Multivariate MLE
for i=1:N
    x = cast(samples(i, :), 'double');
    mu = mu + x;
end
mu = mu / N;

for i=1:N
    x = cast(samples(i, :), 'double');
    c = x - mu;
    sigma = sigma + (c' * c);
end
sigma = sigma / N;

save("multivariate_params.mat", "mu", "sigma", '-mat');

end