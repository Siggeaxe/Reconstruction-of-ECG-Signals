This project is centered around the 2010 PhysioNet Challenge, titled "Mind the Gap," and specifically focuses on reconstructing missing segments in an ECG signal through adaptive filtering. Two adaptive filtering methods, namely RLS and ADAM, were employed. The project included hyperparameter studies aimed at identifying optimal filter parameters for both methods.

The filters underwent training using 10-minute recordings of two distinct heart signals: ECG V and ECG AVR. To determine hyperparameters such as filter length (Adam & RLS) and forgetting factor (RLS), grid search methodology was applied. The search began with a broad range of hyperparameter values, progressively narrowing the span with an equivalent number of hyperparameter values. This approach was adopted to enhance the resolution of the search.

The missing 10 seconds of the ECG signals were subsequently reconstructed using the two adaptive filters and evaluated using two quality functions derived from the PhysioNet challenge:
- Q1 = 1 - mse(x[n], x_hat[n]) / var(x[n])
- Q2 = cov(x[n], x_hat[n]) / sqrt(var(x[n]) * var(x_hat[n]))

The average scores obtained were as follows:
- RLS: Q1=0.954, Q2=0.979
- ADAM: Q1=0.926, Q2=0.962

Drawing conclusions from the results, it is evident that the two adaptive filtering methods differ. RLS demonstrated superior ability to reconstruct the missing ECG signal, albeit with an associated increase in processing time due to the algorithm's complexity (O(p^2) for RLS vs. O(p) for LMS, where p is the filter length).
