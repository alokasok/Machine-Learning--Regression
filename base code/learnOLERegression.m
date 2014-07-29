function w = learnOLERegression(X,y)

% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1

temp1 = transpose(X)*X;
temp2 = inv(temp1)*transpose(X);
w = temp2*y;
%disp(w);