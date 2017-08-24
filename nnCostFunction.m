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
        ghX_i = ghX(:,c);
    
    % Compute total cost of all (K*m) error values
        J = J + (-1/m) * sum(y_i .* log(ghX_i) + (1-y_i) .* log(1-ghX_i));
end

%%% REGULARIZE THE COST %%
% Outer sum from 1:size(Theta"X+1", 1), inner sum from 1:size(Theta"X", 1)
% (:,2:end) because 1st term is bias unit, which doesn't get regularized
    J = J + (lambda/(2*m))*(sum( sum(Theta1(:,2:end).^2) ) + ...
                            sum( sum(Theta2(:,2:end).^2) ));
                    
%% =============== Part 2: Implement Backpropagation ===============

for t = 1:m
    % Set the input layer's values (a_1) to the t-th training example x(t).
        a_1 = [1; X(t,:)'];
    % Perform a feedforward pass computing
    % the activations (z_2, a_2, z_3, a_3)
        z_2 = Theta1 * a_1;
        a_2 = [1; sigmoid(z_2)];
       
        z_3 = Theta2 * a_2;
        a_3 = sigmoid(z_3);
    
        
    % For each output unit k in layer 3 (the output layer), set
        y_k =(1:K == y(t))';
        delta_3 = a_3 - y_k;
    % where y_k (values == 0 or 1) indicates whether the current training 
    % example belongs to class k (y_k == 1) or not (y_k == 0)
    
    
    % For the hidden layer l = 2, set
        delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z_2)];
    
        
    % Accumulate the gradient from this example using the following
    % formula. Note that you should skip or remove delta_2(0)
        delta_2 = delta_2(2:end);
        Theta1_grad = Theta1_grad + delta_2 * a_1';
        
        Theta2_grad = Theta2_grad + delta_3 * a_2';
        
end

        
% Obtain the (unregularized) gradient for the neural network cost
% function by dividing the accumulated gradients by 1/m
    Theta1_grad = Theta1_grad/m;
    Theta2_grad = Theta2_grad/m;

% Add regularization to the gradient (excludes Theta(1) (i.e. theta_0)
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ...
        lambda/m * Theta1(:,2:end);
    
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ...
        lambda/m * Theta2(:,2:end);

% Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
