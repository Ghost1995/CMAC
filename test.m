function [ accuracy ] = test( map, testData, state )
% This function tests the accuracy of the trained neural network according
% to the testing data.

% map is the CMAC map created using the create function. It consists of the
% input vector, the look up table, the corrected weights, and the number of
% association cells linked to each input vector.
% testData is the data to be used to test the trained CMAC.

% tic

if isempty(map) || isempty(testData)
    accuracy = NaN;
    return
end

% define location of input w.r.t. input vectors
input  = zeros(length(testData),2);
for i=1:length(testData)
    if testData(i,1) > map{1}(end)
        input(i,1) = length(map{1});
    elseif testData(i,1) < map{1}(1)
        input(i,1) = 1;
    else
        temp = (length(map{1})-1)*(testData(i,1)-map{1}(1))/(map{1}(end)-map{1}(1)) + 1;
        input(i,1) = floor(temp);
        if (ceil(temp) ~= floor(temp)) && state
            input(i,2) = ceil(temp);
        end
    end
end

% compute accuracy
numerator = 0;
denominator = 0;
for i=1:length(input)
    if input(i,2) == 0
        output = sum(map{3}(find(map{2}(input(i,1),:))));
        numerator = numerator + abs(testData(i,2)-output);
        denominator = denominator + testData(i,2) + output;
    else
        d1 = norm(map{1}(input(i,1))-testData(i,1));
        d2 = norm(map{2}(input(i,2))-testData(i,1));
        output = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
               + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
        numerator = numerator + abs(testData(i,2)-output);
        denominator = denominator + testData(i,2) + output;
    end
    Y(i) = output;
end
error = abs(numerator/denominator);
accuracy = 100 - error;

[X,I] = sort(testData(:,1));
Y = Y(I);
plot(X,Y);

% toc

end