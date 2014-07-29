% load the data
load diabetes;
x_train_i = [ones(size(x_train,1),1) x_train];
x_test_i = [ones(size(x_test,1),1) x_test];


%%% FILL CODE FOR PROBLEM 1 %%%
weight = learnOLERegression(x_train,y_train);
value = sum((y_train - (x_train*weight)).^2);
error_train = sqrt(value);
%weight = learnOLERegression(x_test,y_test);
value = sum((y_test - (x_test*weight)).^2);
error_test = sqrt(value);
weight = learnOLERegression(x_train_i,y_train);
value = sum((y_train - (x_train_i*weight)).^2);
error_train_i = sqrt(value);
%weight = learnOLERegression(x_test_i,y_test);
value = sum((y_test - (x_test_i*weight)).^2);
error_test_i = sqrt(value);
display(error_train_i);
display(error_test_i);
display(weight);


%%% END PROBLEM 1 CODE %%%


%%% FILL CODE FOR PROBLEM 2 %%%
% ridge regression using least squares - minimization
%lambdas = 0:0.001:0.5;
lambdas = 0:0.00001:0.001;

train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);
%test_errors = display(error_train);
display(error_test);
% linear regression with intercept
zeros(length(lambdas),1);

for i = 1:length(lambdas)
    lambda = lambdas(i);
    weight = learnRidgeRegression(x_train_i,y_train,lambda);
    train_errors(i) = sqrt(sum((y_train - (x_train_i*weight)).^2));
    test_errors(i) = sqrt(sum((y_test - (x_test_i*weight)).^2));
    if(lambda == 0.0002)
        display(lambda);
        display(train_errors(i));
        display(test_errors(i));
    end
    if(lambda == 0.0004)
        display(lambda);
        display(train_errors(i));
        display(test_errors(i));
    end
    if(lambda == 0.0008)
        display(lambda);
        display(train_errors(i));
        display(test_errors(i));
    end
    if(lambda == 0.001)
        display(lambda);
        display(train_errors(i));
        display(test_errors(i));
    end
    % fill code here for prediction and computing errors
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
[D P] = min(test_errors);
lambda_optimal = lambdas(P);
%%% END PROBLEM 2 CODE %%%

%%% BEGIN PROBLEM 3 CODE
% ridge regression using gradient descent - see handouts (lecture 21 p5) or
% http://cs229.stanford.edu/notes/cs229-notes1.pdf (page 11)
initialWeights = zeros(65,1);
% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 500);

% define the objective function
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);

% run ridge regression training with fmincg
for i = 1:length(lambdas)
    lambda = lambdas(i);
    objFunction = @(params) regressionObjVal(params, x_train_i, y_train, lambda);
    w = fmincg(objFunction, initialWeights, options);
    % fill code here for prediction and computing errors
    train_errors(i) = sqrt(sum((y_train - (x_train_i*w)).^2));
    test_errors(i) = sqrt(sum((y_test - (x_test_i*w)).^2));
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');

%%% END PROBLEM 3 CODE


%%% BEGIN  PROBLEM 4 CODE
% using variable number 3 only
x_train = x_train(:,3);
x_test = x_test(:,3);
train_errors = zeros(7,1);
test_errors = zeros(7,1);

% no regularization
lambda = 0;
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    train_errors(d+1) = sqrt(sum((y_train - (x_train_n*w)).^2));
    test_errors(d+1) = sqrt(sum((y_test - (x_test_n*w)).^2));
    % fill code here for prediction and computing errors
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');

% optimal regularization
lambda = lambda_optimal; % from part 2
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    train_errors(d+1) = sqrt(sum((y_train - (x_train_n*w)).^2));
    test_errors(d+1) = sqrt(sum((y_test - (x_test_n*w)).^2));
    % fill code here for prediction and computing errors
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
display(lambda_optimal);
