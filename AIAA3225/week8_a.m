global gamma;
global a;
global b;
gamma = 2;
a = 10;
b = 3;

l = gamma + pow2(a) / 4;
n = 1 / l;

function y = f(x)
    global a;
    global b;
    global gamma;
    y = log(1 + exp(a * x)) - b * x + 1/2 * gamma * x * x;
end

function y = df(x)
    global a;
    global b;
    global gamma;
    y = 1 / (1 + exp(a * x)) * exp(a * x) * a - b + gamma * x;
end

epoch = 10;

x0 = -10;

x = zeros(epoch + 1);
y = zeros(epoch + 1);

x(1) = x0;
y(1) = f(x0);

for s = 1 : (epoch - 1)
    x(s + 1) = x(s) - n * df(x(s));
    y(s + 1) = f(x(s + 1));
end

i = [1 : epoch + 1];
plot(i, y, '-r*')



