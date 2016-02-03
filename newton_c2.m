function newton_c2()
clearvars
clc

THEX = 0;
THEY = 0;
CONV = 0;

tol = 10^(-8);
limit = 1000;
nu = 10^(-4);
tau = 0.9;
rho = 0.5;
mu = 100;

x = 6;
y = 5;
x_old = 3;
y_old = 4;
lambda = 1;
lambda_old = 1;
B = eye(2);

p_k = [1;1];
l_k = 1;

k = 0;

while(1)

    k = k + 1;
    if (k == limit)
        break
    end

    epsilon = norm(grad_f(x,y)-lambda*grad_c(x,y));
    
    CONV(k) = epsilon;
    THEX(k) = x;
    THEY(k) = y;
    
    if ( epsilon < tol )
        break
    end
    
    F = [grad_f(x,y) - lambda * grad_c(x,y) ; c(x,y)];
    F_prime = [hess(x,y), -grad_c(x,y) ; grad_c(x,y)', 0];
    P_K = inv(F_prime)*(-F);
    p_k = [P_K(1);P_K(2)];
    l_k = P_K(3);
    
    
    alpha = 1;
    j = 0;
    x_new = x;
    y_new = y;
    while(1)
        j = j+1;
        if (j == 100)
            break
        end
        x_new = x_new + alpha * P_K(1);
        y_new = y_new + alpha * P_K(2);
        phi1 = f(x_new,y_new) + mu*abs(c(x_new,y_new));
        phi0 = f(x,y) + mu * abs(c(x,y));
        D = -(p_k'*grad_f(x,y))-mu*abs(c(x,y));
        RHS = phi0 + nu*alpha*D;
        if (phi1 <= RHS)
            break
        end
        alpha = alpha * tau;
    end  
    x_old = x;
    y_old = y;
    lambda_old = lambda;
    x = x_old + alpha * P_K(1);
    y = y_old + alpha * P_K(2);
    lambda = lambda_old + alpha*P_K(3);
    
    
end
x
y
lambda
k

semilogy(CONV)
xlabel('Number of iterations')
ylabel('Norm of \nabla£_x(x,y)')
title('Figure 6: Newton''s Method: c(x) = 1 - x^2 - y^2')
grid on
size = 8;
a = linspace(-size,size);
b = linspace(-size,size);
[A,B] = meshgrid(a,b);
C = (1-A).^2+100*((B-(A.^2)).^2);
levels = 100:10:10;
figure
hold on
contour(A,B,C,200)
plot(THEX,THEY,'Black')
plot(6,5,'g*')
plot(x,y,'r*')
[C,D]=circle1;
plot(C,D,'c--')
legend('\nabla£_x(x,y)','Path','Start','End','c(x) = 1 - x^2 - y^2','Location','NorthWest')
xlabel('X')
ylabel('Y')
title('Figure 7: Newton''s Method: c(x) = 1 - x^2 - y^2')

dlmwrite('newton_c2.txt',CONV)
end



function f1 = f(x,y)
    f1 = (x - 1)^2 + 100*(- x^2 + y)^2;
end

function grad_f1 = grad_f(x,y)
    grad_f1 = [2*x - 400*x*(- x^2 + y) - 2 ; - 200*x^2 + 200*y];
end

function hess1 = hess(x,y)
    hess1 = [ 1200*x^2 - 400*y + 2, -400*x; -400*x, 200];
end

function c1 = c(x,y)
    c1 = 1 - x^2 - y^2;
end

function grad_c1 = grad_c(x,y)
    grad_c1 = [-2*x ; -2*y];
end

function [C,D] = circle1()
    th = 0:pi/50:2*pi;
    C = cos(th);
    D = sin(th);
end











