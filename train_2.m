function [ map, iteration, finalError, t ] = train_2( parentMap, trainData, E )
% This function trains the neural network according to the training data.
% This implies that this function gives the CMAC architecture with the
% corrected weights.

% map is the CMAC map created using the create function. It consists of the
% input vector, the look up table, the initial weights, and the number of
% association cells linked to each input/output vector.
% trainData is the data to be used to train CMAC.
% E is the acceptable error. This error is in terms of the training data.

tic;

map = parentMap;
if isempty(map) || isempty(trainData) || isempty(E)
    return
end

% define location of input w.r.t. input vectors
input  = zeros(length(trainData),1);
for i=1:length(trainData)
    if trainData(i,1) > map{1}(end)
        input(i) = length(map{1});
    elseif trainData(i,1) < map{1}(1)
        input(i) = 1;
    else
        temp = (length(map{1})-1)*(trainData(i,1)-map{1}(1))/(map{1}(end)-map{1}(1)) + 1;
        input(i) = floor(temp);
    end
end

% compute output for each input and accordingly adjust weights until the
% specified number of iterations is achieved
eta = 0.025; % learning rate
error = Inf;
iteration = 0;
count = 0;
y = zeros(size(map{2},1),1);
while (error > E)&&(2*count <= iteration)
    old_err = error;
    iteration = iteration + 1;
    
    % compute output for each input and accordingly adjust weights
    for i=1:length(input)
        index = find(map{2}(input(i),:));
        output = sum(map{3}(index(1:map{4}))) + (1/(1+exp(-y(input(i)))))*sum(map{3}(index(map{4}+1:end)));
%         for j=1:size(y)
%             index_out = find(map{2}(j,:),map{4},'last');
%             output = output + y(j)*sum(map{3}(index_out));
%         end
%         y(input(i)) = output;
        error = eta*(trainData(i,2)-output)/(2*map{4});
        map{3}(index) = map{3}(index) + error;
%         for j=1:size(y)
%             index_out = find(map{2}(j,:),map{4},'last');
%             map{3}(index_out) = map{3}(index_out) + error;
%         end
    end

    % compute final error
    numerator = 0;
    denominator = 0;
    for i=1:length(input)
        output = sum(map{3}(find(map{2}(input(i,1),:))));
        numerator = numerator + abs(trainData(i,2)-output);
        denominator = denominator + trainData(i,2) + output;
    end
    error = abs(numerator/denominator);
    if abs(old_err - error) < 0.00001
        count = count + 1;
    else
        count = 0;
    end
end
iteration = iteration - count;

% compute final error
numerator = 0;
denominator = 0;
for i=1:length(input)
    Y(i) = sum(map{3}(find(map{2}(input(i,1),:))));
    numerator = numerator + abs(trainData(i,2)-Y(i));
    denominator = denominator + trainData(i,2) + Y(i);
end
finalError = abs(numerator/denominator);
[X,I] = sort(trainData(:,1));
Y = Y(I);
plot(X,Y);

t = toc;

end