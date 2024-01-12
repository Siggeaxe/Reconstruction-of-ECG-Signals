% 1TE651 Signal Processing
% Sigge & Achilles
% Reconstruction of a missing ECG signal with RLS (Recursive Least Squares)
% Hyperparameter study of forgetting factor

% uncomment if needed
clear all
close all
clc

patient_no = 1;
s=string(patient_no);

x_1 = importdata('DATASET/ECG_'+s+'/ECG_'+ s +'_V.mat');
x_2 = importdata('DATASET/ECG_'+s+'/ECG_'+ s +'_AVR.mat');
x_T = importdata('DATASET/ECG_'+s+'/ECG_'+ s +'_II.mat');
x_missing = importdata('DATASET/ECG_'+s+'/ECG_'+ s +'_II_missing.mat');

% Make signals zero-mean
mean1 = mean(x_1);
mean2 = mean(x_2);
meanT = mean(x_T);
x_1 = x_1 - mean1;
x_2 = x_2 - mean2;
x_T = x_T - meanT;

% number of iterations
Ntot = length(x_1);
Nsim = length(x_T);
Nmissing = length(x_missing);

% number of unknowns, length of theta vector
N = 10;
M = 10;
p = N + M;

% RLS forgetting factor
mylambda = linspace(0.5, 1, 100);

q1 = zeros(1, 100);
q2 = zeros(1, 100);

for k = 1:100

    % Observations H
    h = zeros(p, Ntot);
    for i=p:Ntot
         h(:,i) = [x_1(i:-1:i-N+1, 1); x_2(i:-1:i-M+1, 1)];
    end
    
    % Initilizations n = -1
    mytheta=1*ones(p, 1);   % Parameter vector
    P=1*eye(p);             % Covariance of parameters
    
    for i=1:(Nsim)          % set n = 0
        % Prediction error
        e = x_T(i, 1) - h(:,i)'*mytheta;
        
        % update kalman gain (alternative form)
        K = P*h(:,i)/(mylambda(k) + h(:,i)'*P*h(:,i));
        
        % update theta 
        mytheta = mytheta + K*e; 
        
        % update P (alternative form)
        P = mylambda(k)^-1*(eye(p) - K*h(:,i)')*P;
    end
    
    %Calculate estimate
    x_missing_estimate = h'*mytheta;
    x_missing_estimate = x_missing_estimate + meanT;
    x_missing_estimate = x_missing_estimate(Nsim+1:end);
    
    %Evaluation
    q1(k) = Q1(x_missing, x_missing_estimate);
    q2(k) = Q2(x_missing, x_missing_estimate);
end

% Plot results
figure
hold on
plot(mylambda, q1)
plot(mylambda, q2)
legend("Q1", "Q2")
xlabel("Lambda")
ylabel("Q")
title("Hyper Parameter Study")


%%% Functions
% Estimate covariance
function my_cov = my_cov(x, y)
    mx = mean(x);
    my = mean(y);
    my_cov = 0;
    
    for i = 1:1:length(x)
        my_cov = my_cov + (x(i) - mx)*(y(i) - my);
    end
    
    my_cov = my_cov/length(x);
end

% Estimate mse
function my_mse = my_mse(x, y)
    my_mse = 0;
    
    for i = 1:1:length(x)
        my_mse = my_mse + (x(i) - y(i))^2;
    end
    
    my_mse = my_mse/length(x);
end

% Evaluation Q1
function Q1 = Q1(x, x_hat)
    Q1 = 1 - my_mse(x, x_hat)/my_cov(x, x);
    
    if (Q1 < 0)
        Q1 = 0;
    end
end

% Evaluation Q2
function Q2 = Q2(x, x_hat) 
    Q2 = my_cov(x, x_hat)/sqrt(my_cov(x, x)*my_cov(x_hat, x_hat));
    
    if (Q2 < 0)
        Q2 = 0;
    end
end
