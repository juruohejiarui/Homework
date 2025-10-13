cvx_begin sdp
% cvx_solver sedumi
% cvx_precision best
    variable X(3,3);
    target = X(1,2) + X(1,3);
    minimize target
    
    subject to
        X >= 0;
        diag(X).' == [1,1,1];
cvx_end

fprintf('status: %s\n', cvx_status)
X