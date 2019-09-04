clear all;
rng(19);

% Load data
load('data.mat')

%% GRID SEARCH for Training the Network:

% Training Function:
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation

%trainFcn = 'trainscg'; % Scaled Conjugate Gradient backpropagation
                        % We also tryed to use the training function
                        % 'trainscg'. As 'trainlm' produces better results we
                        % decided to work with 'trainlm'.
                        
                        % For trying the 'trainscg' training funcion one must
                        % uncomment the line for 'trainscg' (line 12) and 
                        % comment the line for the 'trainlm' training 
                        % funciton (line 10)

% Hyperparameters:
learningRate = [0.05, 0.2, 0.5, 0.9];   
momentumParam = [0.3, 0.6, 0.8, 0.9];
hiddenLayerSize = [8, 10, 12, 15];

% Table to store the results of the grid search in (average of all 10
% iterations of the cross validation stored)
tableGrid = [];
updateGrid = [];

% Accuracy/Error/whole Table (each value of each of the 10 iterations
% within the Cross Validaiton is included)
AccuracyAll = [];
ErrorAll = []; 
updateTable_CV = [];
Table_CV = [];

counter = 0;

% 10-fold cross validation
indices = crossvalind('Kfold', classes_Training,10);

% Grid Search:
% 1st loop goes throughfour different learning rates 
% 2nd loop iterates through four different numbers for momentum
% 3dr loop goes through different numbers of neurons for the four hidden layers 
% With a 10-Fold Cross-Validation --> all the hyperparameters are trained
% 10 times, on 9 different training sets and tested on changing 1 validation set
for i = learningRate
    
    for j = momentumParam
        
      for k = hiddenLayerSize
          
          Accuracy = [];
          Error = [];
          counter = counter+1;
          
        for x = 1:10
           
            % index for validation
            validation_idx = (indices == x);
            train_idx = ~validation_idx;
           
            % network with four hidden layers:
            net = fitnet([k,k,k,k],trainFcn);
            net.trainParam.epochs = 200;  % Max. number of epochs (stopps after 200 epochs)
            
            
          
            
            %net.trainParam.max_fail = 100     % We tested a different number for the validation checks.
                                               % By increasing it to 100 from the default value of 6 to 
                                               % see if accuracy would improve. 
                                               % As the runtime for the code was almost four times slower
                                               % and as the results were the same, the default value 
                                               % of 6 validation checks was maintained. 
            
            net.performFcn = 'mse';       % default Error function

            net.trainParam.lr = i;        % Learning rate 
            net.trainParam.mc = j;        % Momentum parameter 
            net.layers{1}.transferFcn = 'tansig';   % Activation Function for the four Layers
            net.layers{2}.transferFcn = 'tansig';
            net.layers{3}.transferFcn = 'tansig';   
            net.layers{4}.transferFcn = 'tansig';

            % training the network
            [net,tr] = train(net,features_Training(train_idx,:)', classes_Training(train_idx,:)');
           
            % y = output of the network
            y = net(features_Training(validation_idx,:)');
           
            % performance of the network (error)
            performance = perform(net,y,classes_Training(validation_idx,:)');

            fprintf('fitnet, performance: %f\n', performance);
            fprintf('number of epochs: %d, stop: %s\n', tr.num_epochs, tr.stop);

            %create the confusion matrix    
            CM_model= confusionmat(classes_Training(validation_idx,:)',round(y));
            
            % calculate the accuracy of the model using the confusion matrix 
            Accuracy_model = 100*sum(diag(CM_model))./sum(CM_model(:));
           
           
            % store the model accuracy values in an array
            Accuracy = [Accuracy; Accuracy_model]; 
            AccuracyAll = [AccuracyAll;Accuracy_model];
           
            % store the model performance values in an array
            Error = [Error; performance];
            ErrorAll =[ErrorAll; performance];

            updateTable_CV = [counter i j k performance Accuracy_model];
            Table_CV = [Table_CV ; updateTable_CV];
           
        end
           
           meanAccuracy = sum(Accuracy)/10;
           meanError = sum(Error)/10;
           updateGrid = [counter i j k meanError meanAccuracy]
           tableGrid = [tableGrid ; updateGrid]
           
      end
      
    end
    
end

%% Finding best hyperparameters
% Index with highest Accuracy in tableGrid
[~,ModelIdx1]=max(tableGrid(:,6));

% Highest Accuracy
max_Accuracy = tableGrid(ModelIdx1,6);
ValuesMaxAccuracy = tableGrid(ModelIdx1,:);

% Index with lowest Error in tableGrid
[~,ModelIdx2]=min(tableGrid(:,5));

% Lowest Error
min_Error = tableGrid(ModelIdx2,5);
ValuesMinError = tableGrid(ModelIdx2,:);

% Find the best parameters that result in the highest Accuracy
ValuesMaxAccuracy = array2table(ValuesMaxAccuracy);
BestLearningRate = ValuesMaxAccuracy{:,2} ; 
BestMomentumParameter = ValuesMaxAccuracy{:,3};
BestHiddenLayerSize = ValuesMaxAccuracy{:,4};



%% BEST MODEL:
% Train a new model using the "best" parameters 
netBest = fitnet([BestHiddenLayerSize,BestHiddenLayerSize,BestHiddenLayerSize,BestHiddenLayerSize],trainFcn);
netBest.trainParam.epochs = 200;               % Num. epochs (after 200 epochs the training stops)
netBest.performFcn = 'mse';                    % Error function
netBest.trainParam.lr = BestLearningRate;      % Learning rate 
netBest.trainParam.mc = BestMomentumParameter; % Momentum parameter

% Train network with the training data
[netBest,tr2] = train(netBest,features_Training', classes_Training');


%% TRAINING ACCURACY
% Test network with the training data in order to get the training accuracy (training outcome)
y_Training = netBest(features_Training');

% Performance Ttraining
performanceTraining = perform(netBest,y_Training,classes_Training');

% Confustion matrix training
trainingConfusion= confusionmat(classes_Training',round(y_Training));

% Accuracy Training
trainingAccuracy = 100*sum(diag(trainingConfusion))./sum(trainingConfusion(:));

figure(1);
confusionchart(trainingConfusion)
title("MLP Confusion Matrix of Training Data")

% Training Metrics - Precision | Recall | F1-Score
% Class 1
training_precision_class1 = trainingConfusion(1,1) / ...
    (trainingConfusion(1,1) + (trainingConfusion(2,1)+trainingConfusion(3,1)))

training_recall_class1 = trainingConfusion(1,1) / ...
    (trainingConfusion(1,1) + (trainingConfusion(1,2)+trainingConfusion(1,3)))

training_F1Score_class1 = 2 * ((training_precision_class1*training_recall_class1)/(training_precision_class1+training_recall_class1))

%Class 2
training_precision_class2 = trainingConfusion(2,2) / ...
    (trainingConfusion(2,2) + (trainingConfusion(1,2)+trainingConfusion(3,2)))

training_recall_class2 = trainingConfusion(2,2) / ...
    (trainingConfusion(2,2) + (trainingConfusion(2,1)+trainingConfusion(2,3)))

training_F1Score_class2 = 2 * ((training_precision_class2*training_recall_class2)/(training_precision_class2+training_recall_class2))

%Class 3
training_precision_class3 = trainingConfusion(3,3) / ...
    (trainingConfusion(3,3) + (trainingConfusion(1,3)+trainingConfusion(2,3)))

training_recall_class3 = trainingConfusion(3,3) / ...
    (trainingConfusion(3,3) + (trainingConfusion(3,1)+trainingConfusion(3,2)))

training_F1Score_class3 = 2 * ((training_precision_class3*training_recall_class3)/(training_precision_class3+training_recall_class3))

%Final Table TRAINING - Precision | Recall | F1-Score
trainingMetrics = [training_precision_class1, training_recall_class1, training_F1Score_class1; training_precision_class2, ...
    training_recall_class2, training_F1Score_class2; training_precision_class3, training_recall_class3, training_F1Score_class3]
trainingMetrics = array2table(trainingMetrics)
trainingMetrics.Properties.VariableNames = {'Precision', 'Recall', 'F1'}



%% TEST ACCURACY
% Test network with the test data in order to get the test accuracy (unseen data)
y_Testing = netBest(features_Testing');

% Performance Test
performanceBest = perform(netBest,y_Testing,classes_Testing');

fprintf('fitnet, performance: %f\n', performanceBest);
fprintf('number of epochs: %d, stop: %s\n', tr2.num_epochs, tr2.stop);

% Confustion matrix test
testingConfusion= confusionmat(classes_Testing',round(y_Testing));
testingAccuracy = 100*sum(diag(testingConfusion))./sum(testingConfusion(:));

% Confustion chart (test data)
figure(2);
confusionchart(testingConfusion)
title("MLP Confusion Matrix of Testing Data")

% Testing Metrics - Precision | Recall | F1-Score

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





%% GRAPHS

% Hyperparameters vs. Accuracy
% Learning rate vs accuracy
figure(3);
% group the different learning rates that were tested
groupsL=findgroups(tableGrid(:,2));
% take max accuracy per learning rate
LR_vs_Accuracy = splitapply(@max,tableGrid(:,6),groupsL);
plot(LR_vs_Accuracy)
title("Learning Rate vs. Accuracy")
xlabel('Learning Rate')
ylabel('Accuracy (%)')

% Momentum vs accuracy
figure(4);
% group the different values for momentum that were tested
groupsM=findgroups(tableGrid(:,3));
% take max accuracy per momentum
M_vs_Accuracy = splitapply(@max,tableGrid(:,6),groupsM);
plot(M_vs_Accuracy)
title("Momentum vs. Accuracy")
xlabel('Momentum')
ylabel('Accuracy (%)')

% Hidden layer size vs accuracy
figure(5);
% group the different values for hidden layer size that were tested
groupsHL=findgroups(tableGrid(:,4));
% take max accuracy per hidden layer size
HL_vs_Accuracy = splitapply(@max,tableGrid(:,6),groupsHL);
plot(HL_vs_Accuracy)
title("Hidden Layer Size vs. Accuracy")
xlabel('Hidden Layer Size')
ylabel('Accuracy (%)')

