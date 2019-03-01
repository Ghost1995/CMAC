function [ map ] = create( X, numWeights, numCell )
% This function links the input vector with the association cells and
% assign all weights to one.

% X denotes the input values
% numWeights denotes the number of weights (hidden vectors) to be used for
% the CMAC algorithm
% numCell denotes the number of Association cells to be linked with each
% input vector (this is used to dicretize the input space as it is a real
% space with infinite values)
% Note that, the numCell option implies that overlap between the
% consecutive input vectors is equal to numCell-1.

if (numCell > numWeights) || (numCell < 1) || (isempty(X))
    map = [];
    return
end

% define input vector
x = linspace(min(X),max(X),numWeights-numCell+1)';

% define look up table
LUT = zeros(length(x),numWeights);
for i=1:length(x)
    LUT(i,i:numCell+i-1) = 1;
end

% define weights
W = ones(numWeights,1);

% define map
map = cell(3,1);
map{1} = x;
map{2} = LUT;
map{3} = W;
map{4} = numCell;

end