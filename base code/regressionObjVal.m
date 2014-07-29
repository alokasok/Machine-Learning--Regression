function [error, error_grad] = regressionObjVal(w, X, y, lambda)

% compute squared error (scalar) and gradient of squared error with respect
% to w (vector) for the given data X and y and the regularization parameter
% lambda
error = sum((y - (X*w)).^2)/size(y,1) + lambda*(transpose(w)*w);
error_grad = 2*(transpose(X)*(X*w-y))/size(y,1) + 2*lambda*w;