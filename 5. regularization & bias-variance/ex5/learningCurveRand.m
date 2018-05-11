function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
r = size(Xval, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for i= 1:m
    j_train = zeros(1,50);
    j_val = zeros(1,50);
    
    for j = 1:50
        a = randperm(m,i);
        b = randperm(r,i);
        X_sel = X(a,:);
        y_sel = y(a);
        X_val_sel = Xval(b,:);
        y_val_sel = yval(b);
        [theta] = trainLinearReg(X_sel, y_sel, 0.01);
        j_train(j) = linearRegCostFunction(X_sel, y_sel, theta, 0);
        j_val(j) = linearRegCostFunction(X_val_sel, y_val_sel, theta, 0);
    end
    
    error_train(i) = sum(j_train) / 50;
    error_val(i) = sum(j_val) / 50;

    
end

% -------------------------------------------------------------

% =========================================================================

end
