A_0 = [1, 0; 0, 1];
A_1 = [-1, 0; 0, 1];
A_2 = [0, 1; 1, 0];
c = [2, 1];
cvx_begin sdp
% cvx_solver sedumi
% cvx_precision best
    variables x y;
    target = c * [x; y];
    minimize target;
    subject to
        (A_0 + x * A_1 + y * A_2) >= 0;
cvx_end

fprintf('optimal value: %s\n', cvx_optval)
fprintf('solution: %f %f\n', x, y)