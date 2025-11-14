M_1 = [2, 1; 1, 3];
M_2 = [1, 0; 0, 4];
M_3 = [3, -1; -1, 2];
I = eye(2)

a = [0, 1, 1, 1];
c = [1, 0, 0, 0];

cvx_begin sdp quiet
    variable x(4)
    
    target = c * x;
    minimize target;
    subject to
        x(1) * I - x(2) * M_1 - x(3) * M_2 - x(4) * M_3 >= 0;
        a * x == 1;
        x(2) >= 0;
        x(3) >= 0;
        x(4) >= 0;
cvx_end

fprintf("cvx status: %s\n", cvx_status)
fprintf("optimal value: %.4f\n", cvx_optval)
fprintf("optimal solution: " )
x