function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% for it=1:m
%    
%     h=sigmoid(X*theta);
%     temp=-y(it)*log(h(it))-(1-y(it))*log(1-h(it));
%     J=J+temp;
%       
% end
% 
% J=1/m*J;

% by vectorization

h=sigmoid(X*theta);

part_1=-y'*log(h);

part_2=-(1-y')*log(1-h);

temp_1=part_1+part_2;

J=J+1/m*temp_1;

temp_2=X'*(h-y);

grad=grad+1/m*temp_2;

% =============================================================

end
