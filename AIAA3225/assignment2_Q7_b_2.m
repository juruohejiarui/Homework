A_0 = [1, 0; 0, 1];
A_1 = [-1, 0; 0, 1];
A_2 = [0, 1; 1, 0];

cvx_begin sdp
% cvx_solver sedumi
% cvx_precision best
    variable Y(2, 2) symmetric;
    target = -trace(Y' * A_0);
    maximize target;
    subject to
        Y >= 0;
        2 - trace(Y' * A_1) == 0;
        1 - trace(Y' * A_2) == 0;
cvx_end

fprintf('optimal value: %s\n', cvx_optval)
fprintf('solution:')
Y