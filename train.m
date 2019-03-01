function [ map, iteration, finalError, t ] = train( parentMap, trainData, E, state )
% This function trains the neural network according to the training data.
% This implies that this function gives the CMAC architecture with the
% corrected weights.

% map is the CMAC map created using the create function. It consists of the
% input vector, the look up table, the initial weights, and the number of
% association cells linked to each input vector.
% trainData is the data to be used to train CMAC.
% E is the acceptable error. This error is in terms of the training data.
% state is the measure which states whether the CMAC is discrete or
% continuous

tic;

map = parentMap;
if isempty(map) || isempty(trainData) || isempty(E)
    return
end

% define location of input w.r.t. input vectors
input  = zeros(length(trainData),2);
for i=1:length(trainData)
    if trainData(i,1) > map{1}(end)
        input(i,1) = length(map{1});
    elseif trainData(i,1) < map{1}(1)
        input(i,1) = 1;
    else
        temp = (length(map{1})-1)*(trainData(i,1)-map{1}(1))/(map{1}(end)-map{1}(1)) + 1;
        input(i,1) = floor(temp);
        if (ceil(temp) ~= floor(temp)) && state
            input(i,2) = ceil(temp);
        end
    end
end

% compute output for each input and accordingly adjust weights until the
% specified number of iterations is achieved
eta = 0.025; % learning rate
error = Inf;
iteration = 0;
count = 0;
while (error > E)&&(2*count <= iteration)
    old_err = error;
    iteration = iteration + 1;
    
    % compute output for each input and accordingly adjust weights
    for i=1:length(input)
        if input(i,2) == 0
            output = sum(map{3}(find(map{2}(input(i,1),:))));
            error = eta*(trainData(i,2)-output)/map{4};
            map{3}(find(map{2}(input(i,1),:))) = map{3}(find(map{2}(input(i,1),:))) + error;
        else
            d1 = norm(map{1}(input(i,1))-trainData(i,1));
            d2 = norm(map{1}(input(i,2))-trainData(i,1));
            output = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
                    + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
            error = eta*(trainData(i,2)-output)/map{4};
            map{3}(find(map{2}(input(i,1),:))) = map{3}(find(map{2}(input(i,1),:)))...
                                                    + (d2/(d1+d2))*error;
            map{3}(find(map{2}(input(i,2),:))) = map{3}(find(map{2}(input(i,2),:)))...
                                                    + (d1/(d1+d2))*error;            
        end
    end

    % compute final error
    numerator = 0;
    denominator = 0;
    for i=1:length(input)
        if input(i,2) == 0
            output = sum(map{3}(find(map{2}(input(i,1),:))));
            numerator = numerator + abs(trainData(i,2)-output);
            denominator = denominator + trainData(i,2) + output;
        else
            d1 = norm(map{1}(input(i,1))-trainData(i,1));
            d2 = norm(map{1}(input(i,2))-trainData(i,1));
            output = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
                   + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
            numerator = numerator + abs(trainData(i,2)-output);
            denominator = denominator + trainData(i,2) + output;
        end
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
    if input(i,2) == 0
        Y(i) = sum(map{3}(find(map{2}(input(i,1),:))));
        numerator = numerator + abs(trainData(i,2)-Y(i));
        denominator = denominator + trainData(i,2) + Y(i);
    else
        d1 = norm(map{1}(input(i,1))-trainData(i,1));
        d2 = norm(map{1}(input(i,2))-trainData(i,1));
        Y(i) = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
               + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
        numerator = numerator + abs(trainData(i,2)-Y(i));
        denominator = denominator + trainData(i,2) + Y(i);
    end
end
finalError = abs(numerator/denominator);
[X,I] = sort(trainData(:,1));
Y = Y(I);
% plot(X,Y);

t = toc;

end