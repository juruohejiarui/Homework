
% 
D = zeros(6, 6);
A = zeros(6, 6);
E = [1,2;1,3;1,4;2,3;2,5;3,6;4,5;4,6;5,6];
m = size(E, 1);
for i = 1 : m
    x = E(i,1);
    y = E(i,2);
    D(x, x) = D(x, x) + 1;
    D(y, y) = D(y, y) + 1;
    A(x, y) = A(x, y) + 1;
    A(y, x) = A(y, x) + 1;
end
A
D
L_G = D - A
cvx_begin sdp
% cvx_precision best
    variable X(n, n) symmetric;
    target = 1/4 * trace(L_G * X);
    maximize target;
    subject to
        X >= 0;
        diag(X) == ones(n, 1);
cvx_end

[V, D] = eig(X);
V = V * (D^(1/2));
V