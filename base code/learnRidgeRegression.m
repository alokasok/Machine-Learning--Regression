function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1

temp1 = transpose(X)*X;
column_size = size(X,2);
row_size = size(X,1);
lambda_matrix = row_size*lambda * eye(column_size);
temp2 = inv(temp1 + lambda_matrix);
w = (temp2 * transpose(X))*y;

