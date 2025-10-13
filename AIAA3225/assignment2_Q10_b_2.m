rho = [1/3, 1/10, 0; 1/10, 1/3, 0; 0, 0, 1/3];
sigma = diag([2/5, 1/4, 7/20]);

tmp = eig(rho - sigma);

fprintf("currect optimal value is: %.4f\n", ...
    1/2 * (1 - 1/2 * sum(abs(tmp))));

cvx_begin sdp quiet
% cvx_precision best
    variable B(3, 3) hermitian;
    target = 1/2 - trace(B' * eye(3));
    maximize target;
    subject to
        B >= 0;
        B >= 1/2 * (rho - sigma);
cvx_end

fprintf("cvx status: %s\n", cvx_status)
fprintf("optimal value: %.4f\n", cvx_optval)
fprintf("optimal solution: ")
B