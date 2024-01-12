% 1TE651 Signal Processing
% Sigge & Achilles
% Reconstruction of a missing ECG signal with ADAM algorithm
% Hyperparameter study of filter length with grid search

% uncomment if needed
clear all
close all
clc

patient_no = 1; % choose between patients: 1, 2, ..., 8
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

% Filter length, p = N + M
N = 30;
M = 30;

% Saving the evaluations scores from the grid search
q1_saved = zeros(N, M);
q2_saved = zeros(N, M);
q_avg_saved = zeros(N, M);

% Adam parameters
alfa = 0.001;   % Stepsize
beta1 = 0.9;    % Exponential decay rates for the FIRST moment estimates
beta2 = 0.999;  % Exponential decay rates for the SECOND moment estimates
epsilon = 1e-8; % For numerical stability??

% Hyperparameter Filter Length
for M = 1:M
    parfor N = 1:N  % Parallel computing toolbox
        p = N + M;
   
        % Observations y
        y = zeros(p, Ntot);
        for n=p:Ntot
            y(:,n) = [x_1(n:-1:n-(M-1)); x_2(n:-1:n-(N-1))];
        end

        % Initilizations n = 0
        h = zeros(p, 1);  % Initial parameter vector
        m = zeros(p, 1);  % Initialize 1st moment vector
        v = zeros(p, 1);  % Initialize 2nd moment vector

        % Adam optimizer algorithm
        for n=1:(Nsim)      % set n = 1
            % Prediction error
            e = x_T(n)-h'*y(:,n);

            % Gradient of f(h[n-1])
            g = -e*y(:,n);

            % Update biased first moment estimate
            m = beta1*m + (1-beta1)*g;

            % Update biased second raw moment estimate
            v = beta2*v+(1-beta2)*g.^2;

            % Compute bias-corrected first moment estimate
            m_hat = m/(1-beta1^n);

            % Compute bias-corrected second raw moment estimate
            v_hat = v/(1-beta1^n);

            % Update parameters
            h = h - alfa*m_hat./(sqrt(v_hat) + epsilon);

        end

    %Calculate estimate
    x_missing_estimate = h'*y;
    x_missing_estimate = x_missing_estimate + meanT;
    x_missing_estimate = x_missing_estimate(Nsim+1:end);

    %Evaluation
    q1 = Q1(x_missing, x_missing_estimate);
    q2 = Q2(x_missing, x_missing_estimate);
    
    q1_saved(N, M) = q1;
    q2_saved(N, M) = q2;
    q_avg_saved(N, M) = (q1 + q2)/2;
    
    end
    
    disp("Iteration M = " + string(M))
end

% Plot
figure
surf(q1_saved)
xlabel("ECG V parameters")
ylabel("ECG AVR parameters")
zlabel("Q1 evaluation")
title("Hyperparameter study of filter length, " + "patient " + s)
colorbar

figure
surf(q2_saved)
xlabel("ECG V parameters")
ylabel("ECG AVR parameters")
zlabel("Q2 evaluation")
title("Hyperparameter study of filter length, " + "patient " + s)
colorbar()

figure
surf(q_avg_saved)
xlabel("ECG V parameters")
ylabel("ECG AVR parameters")
zlabel("Q average evaluation")
title("Hyperparameter study of filter length, " + "patient " + s)
colorbar

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