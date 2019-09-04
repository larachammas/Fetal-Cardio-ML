% --> For testing the final model (Script includes the final model trained 
% with the and the best hyperparameters and precision, recall and 
% F1-Score for training and testing)

clear all 
close all
clc

% Load data
load('data.mat')    

%% BEST MODEL:
% Train a new model using the "best" parameters found during the grid search 
netBest = fitnet([15,15,15,15],'trainlm');     
netBest.trainParam.epochs = 200;               % Num. epochs (after 200 epochs the training stops)
netBest.performFcn = 'mse';                    % Error function
netBest.trainParam.lr = 0.05;                  % Best Learning rate 
netBest.trainParam.mc = 0.9;                   % Best Momentum parameter

% Train network with the training data
[netBest,tr2] = train(netBest,features_Training', classes_Training');


%% TEST ACCURACY
% Test network with the unseen test data in order to get the test accuracy (unseen data)
y_Test = netBest(features_Testing');

% Performance Test
performanceBest = perform(netBest,y_Test,classes_Testing');

% Confustion matrix test
testingConfusion= confusionmat(classes_Testing',round(y_Test));
testingAccuracy = 100*sum(diag(testingConfusion))./sum(testingConfusion(:));

% Confustion chart (test data)
figure(1);
confusionchart(testingConfusion)
title("MLP Confusion Matrix of Testing Data")

%%% Testing Metrics - Precision | Recall | F1-Score

%Class 1
testing_precision_class1 = testingConfusion(1,1) / ...
    (testingConfusion(1,1) + (testingConfusion(2,1)+testingConfusion(3,1)))

testing_recall_class1 = testingConfusion(1,1) / ...
    (testingConfusion(1,1) + (testingConfusion(1,2)+testingConfusion(1,3)))

testing_F1Score_class1 = 2 * ((testing_precision_class1*testing_recall_class1)/ ...
    (testing_precision_class1+testing_recall_class1))

% Class 2
testing_precision_class2 = testingConfusion(2,2) / ...
    (testingConfusion(2,2) + (testingConfusion(1,2)+testingConfusion(3,2)))

testing_recall_class2 = testingConfusion(2,2) / ...
    (testingConfusion(2,2) + (testingConfusion(2,1)+testingConfusion(2,3)))

testing_F1Score_class2 = 2 * ((testing_precision_class2*testing_recall_class2)/...
    (testing_precision_class2+testing_recall_class2))

% Class 3
testing_precision_class3 = testingConfusion(3,3) / ...
    (testingConfusion(3,3) + (testingConfusion(1,3)+testingConfusion(2,3)))

testing_recall_class3 = testingConfusion(3,3) / ...
    (testingConfusion(3,3) + (testingConfusion(3,1)+testingConfusion(3,2)))

testing_F1Score_class3 = 2 * ((testing_precision_class3*testing_recall_class3)/...
    (testing_precision_class3+testing_recall_class3))

%Final Table TESTING - Precision | Recall | F1-Score
testingMetrics = [testing_precision_class1, testing_recall_class1, testing_F1Score_class1; testing_precision_class2, ...
    testing_recall_class2, testing_F1Score_class2; testing_precision_class3, testing_recall_class3, testing_F1Score_class3]
testingMetrics = array2table(testingMetrics)
testingMetrics.Properties.VariableNames = {'Precision', 'Recall', 'F1'}

