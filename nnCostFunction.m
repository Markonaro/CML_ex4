function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % number of training examples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% ================ Part 1: Compute Cost (Feedforward) ================

% Rename num_labels to make code more concise & consistent w/ lectures
K = num_labels;


%%% MAPPING FROM INPUT LAYER A1 TO 1ST HIDDEN LAYER A2 %%%
% Add bias units to all examples in training set
a1 = [ones(m,1), X];
% Determine output of applying weights Theta1 to input layer a1
z2 = a1*Theta1';
% Determine logit of values in hidden layer
a2 = sigmoid(z2);


%%% MAPPING FROM 1ST HIDDEN LAYER A2 TO 2ND HIDDEN LAYER A3 %%%
% Add bias units to all derived values in second layer of NN
a2 = [ones(m,1), a2];
% Determine output of applying weights Theta2 to hidden layer a2
z3 = a2*Theta2';
% Determine logit of values in output layer a3 (i.e. ghX)
ghX = sigmoid(z3);


%%% COMPUTE ERROR FOR EACH EXAMPLE AGAINST EVERY CLASS C %%%
for c = 1:K
    % Generate a vector with ones where y == c and zeros where y != c
    y_i = y(:,1) == c;
    
    % Focus on errors of all 500 examples of class c
    ghX_i = ghX(:, c);
    
    % Compute total cost of all (K*m) error values
    J = J + (-1/m)*sum(y_i.*log(ghX_i)+(1-y_i).*log(1-ghX_i));
end

%%% REGULARIZE THE COST %%
% Outer sum from 1:size(Theta"X+1", 1), inner sum from 1:size(Theta"X", 1)
% (:,2:end) because 1st term is bias unit, which doesn't get regularized
J = J + (lambda/(2*m))*(sum( sum(Theta1(:,2:end).^2) ) + ...
                        sum( sum(Theta2(:,2:end).^2) ));
                    
%% =============== Part 2: Implement Backpropagation ===============

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
