function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vector = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vector = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
error = zeros(length(C_vector),length(sigma_vector));
for i = 1 : length(C_vector)
  for j = 1: length(sigma_vector)
    model= svmTrain(X, y, C_vector(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vector(j)));
    predictions = svmPredict(model, Xval);
    error(i,j) = mean(double(predictions ~= yval));
  end
end

[C_min, C_index] = min(error,[],1);
[error_min, error_index] = min(C_min);  

C = C_vector(C_index(error_index));
sigma = sigma_vector(error_index);


% =========================================================================

end
