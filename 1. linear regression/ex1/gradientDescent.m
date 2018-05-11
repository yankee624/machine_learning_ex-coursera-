function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    delta = (1/m) * (X' * (X * theta - y)); %vectorization 이용하여 theta 한번에 update
    theta = theta - alpha * delta;
    

    %vectorization 없이 theta 각각 update 하려면 이렇게..
%     temp1 = (X * theta - y)' * X(:,1);
%     temp2 = (X * theta - y)' * X(:,2);
%     theta(1) = theta(1) - alpha * (1/m) * temp1;
%     theta(2) = theta(2) - alpha * (1/m) * temp2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
