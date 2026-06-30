function y = mvgpdf(X, mu, sigma)
    N = size(X, 1);
    D = size(mu, 2);
    denominator = sqrt(power(2 * pi, D) * det(sigma));

    y = zeros(N, 1);
    sigma_inv = inv(sigma);

    for i=1:N
        x = X(i, :) - mu;
        exp_term = -(x * sigma_inv * x') / 2.0;
        y(i) = exp(exp_term) / denominator;
    end
end