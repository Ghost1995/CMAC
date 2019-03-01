function [ accuracy ] = test_2( map, testData )
% This function tests the accuracy of the trained neural network according
% to the testing data.

% map is the CMAC map created using the create function. It consists of the
% input vector, the look up table, the corrected weights, and the number of
% association cells linked to each input vector.
% testData is the data to be used to test the trained CMAC.

if isempty(map) || isempty(testData)
    return
end

% define location of input w.r.t. input vectors
input  = zeros(length(testData),1);
for i=1:length(testData)
    if testData(i,1) > map{1}(end)
        input(i) = length(map{1});
    elseif testData(i,1) < map{1}(1)
        input(i) = 1;
    else
        temp = (length(map{1})-1)*(testData(i,1)-map{1}(1))/(map{1}(end)-map{1}(1)) + 1;
        input(i) = floor(temp);
    end
end

% compute accuracy
numerator = 0;
denominator = 0;
for i=1:length(input)
    Y(i) = sum(map{3}(find(map{2}(input(i),:),map{4})));
    numerator = numerator + abs(testData(i,2)-Y(i));
    denominator = denominator + testData(i,2) + Y(i);
end
error = abs(numerator/denominator);
accuracy = 100 - error;

[X,I] = sort(testData(:,1));
Y = Y(I);
plot(X,Y);

end