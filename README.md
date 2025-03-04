# Reconstruction of ECG Signals With Adaptive Filtering

This project is centered around the 2010 PhysioNet Challenge, titled "Mind the Gap," and specifically focuses on reconstructing missing segments in an ECG signal through adaptive filtering. Two adaptive filtering methods, RLS and ADAM, were employed. The project included hyperparameter studies aimed at identifying optimal filter parameters for both methods.

The filters underwent training using 10-minute recordings of two distinct heart signals: ECG V and ECG AVR. To determine hyperparameters such as filter length (Adam & RLS) and forgetting factor (RLS), grid search methodology was applied. The search began with a broad range of hyperparameter values, progressively narrowing the span (with an equivalent number of hyperparameter values). This approach was adopted to enhance the resolution of the search.

| Reconstruction using ADAM (patient 5) | Hyperparameter study (patient 5)  |                                                                                                                                                   
| ---           | ---           |                                                                                                                                                   
| ![](https://github.com/Siggeaxe/Reconstruction-of-ECG-Signals/blob/main/Plots/Adam/Plots/Adam_patient_5_length_15_small.png)  |  ![](https://github.com/Siggeaxe/Reconstruction-of-ECG-Signals/blob/main/Plots/Adam/Hyperpar/Adam_hyper_patient_5_length_30_Q1.png) |


The missing 10 seconds of the ECG signals were subsequently reconstructed using the two adaptive filters and evaluated using two quality functions from the PhysioNet challenge:
- Q1 = 1 - mse(x[n], x_hat[n]) / var(x[n])
- Q2 = cov(x[n], x_hat[n]) / sqrt(var(x[n]) * var(x_hat[n]))

The average scores obtained were as follows:
- RLS: Q1=0.954, Q2=0.979
- ADAM: Q1=0.926, Q2=0.962

The analysis of the results leads to the conclusion that RLS excelled in reconstructing the missing ECG signal. However, this improvement comes at the expense of increased processing time, attributed to the algorithm's complexity (O(p^2) for RLS compared to O(p) for LMS, where p represents the filter length).
