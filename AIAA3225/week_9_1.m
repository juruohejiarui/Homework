function y = getRotMat(theta)
    y = [cos(theta), -sin(theta); sin(theta), cos(theta)];
end
global Q
M = [100, 0; 0, 1];
Q = getRotMat(35/360 * 2 * pi) * M;
function y = f(x)
    global Q;
    y = 1/2 * x * Q * x';
end

function y = gd(x)
    global Q;
    y = Q * x';
end

lambda_max = max(svd(M));
lambda_min = min(svd(M));

kappa = lambda_max / lambda_min;

x_opt = [0, 0];

x_st = [10, 10];

num_epochs = 200;

x_gd = zeros([num_epochs, 2]);
x_nag = zeros([num_epochs, 2]);

bound_gd = zeros([num_epochs, 2]);
bound_nag = zeros([num_epochs, 2]);

x_gd(1, :) = x_st;
x_nag(1, :) = x_st;

beta = sqrt(kappa - 1) / sqrt(kappa + 1);
lr = 1 / lambda_max;

for s = 2 : num_epochs
    d1 = gd(x_gd(s - 1, :))';
    x_gd(s, :) = x_gd(s - 1, :) - lr * d1;
end

for s = 1 : num_epochs
    bound_gd(s, :) = lambda_max / 2 * power(1 - 1 / kappa, s - 1) * (x_gd(s) * x_gd(s)');
end

for s = 2 : num_epochs
    if s == 2
        y = x_nag(s - 1);
    else
        y = x_nag(s - 1) + beta * (x_nag(s - 1) - x_nag(s - 2));
    end
    d2 = gd(y);
    x_nag(s, :) = y - lr * d1;
end

for s = 1 : num_epochs
    bound_nag(s, :) = lambda_max / 2 * power(1 - 1 / sqrt(kappa), s - 1) * (x_nag(s) * x_nag(s)');
end

x = 1 : num_epochs;
semilogy(x, diag(f(x_gd)));
hold on
semilogy(x, diag(f(x_nag)));
semilogy(x, bound_gd);
semilogy(x, bound_nag);
hold off
