clear all
close all
clc

X = (linspace(-5,5))';
Y = abs(X).*sin(X);

trainData = [X(1:70),Y(1:70)];
testData = [X(71:100),Y(71:100)];
CMAC_map = create_2(X,35,10);
figure
plot(X,Y);
hold on
[map,iter,~,T] = train_2(CMAC_map,trainData,0);
acc = test_2(map,testData);
% iteration = zeros(2,34);
% iter = zeros(2,34);
% accuracy = zeros(2,34);
% acc = zeros(2,34);
% t = zeros(2,34);
% T = zeros(2,34);
% for j=1:100
    I = randperm(100);
    trainData = [X(I(1:70)),Y(I(1:70))];
    testData = [X(I(71:100)),Y(I(71:100))];
    for i=1%:34
        CMAC_map = create(X,35,i);
        figure
        plot(X,Y);
        hold on
        [map,iter(1,i),~,T(1,i)] = train(CMAC_map,trainData,0,0);
        acc(1,i) = test(map,testData,0);
        hold off
        legend('Original Curve','Testing Data');
        title(['numCell = ' num2str(i)]);
        figure
        plot(X,Y);
        hold on
        [map,iter(2,i),~,T(2,i)] = train(CMAC_map,trainData,0,1);
        acc(2,i) = test(map,testData,1);
        hold off
        legend('Original Curve','Testing Data');
        title(['numCell = ' num2str(i)]);
    end
%     iteration = iteration + iter;
%     accuracy = accuracy + acc;
%     t = t + T;
% end
% iteration = iteration/j;
% accuracy = accuracy/j;
% t = t/j;