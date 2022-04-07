noise_stddev = 0.01:0.01:0.15;
RMSE_translation = [0.138417, 0.27876, 0.421305, 0.566356, 0.714245, 0.865336, 1.02003, 1.17878, 1.34208, 1.51049, 1.68464, 1.86525, 2.05312, 2.2492, 2.45455];
singular_value_ratio = [0.000432682, 0.0017419, 0.00394312, 0.00705016, 0.0110752, 0.0160288, 0.0219199, 0.0287561, 0.0365431, 0.0452853, 0.0549853, 0.0656441, 0.0772609, 0.0898331, 0.103356];

figure(1);
subplot(2,1,1);
plot(noise_stddev, RMSE_translation, 'r');
ylabel('RMSE of translation');
xlabel('noise stddev');
grid on;
hold on;
subplot(2,1,2);
plot(noise_stddev, singular_value_ratio, 'g');
ylabel('singular value ratio');
xlabel('noise stddev');
grid on;

figure(2);
observe_frames = 1:1:20;      
RMSE_translation_2 = [4.17981, 1.46219, 0.642021, 0.129756, 0.0336155, 0.622189, 0.714245, 0.366828, 0.216166, 0.311195, 0.282865, 0.321004, 0.16281, 0.233319, 0.24987, 0.0659318, 0.034116, 0.0572544, 0.0403843, 0.183698];
singular_value_ratio_2 = [0.108942, 0.00130044, 0.00383875, 0.00575744, 0.00665484, 0.0100492, 0.0110752, 0.0101318, 0.00968966, 0.0131627, 0.0120262, 0.0147766, 0.0139397, 0.0124138, 0.0112787, 0.0130432, 0.0121015, 0.0135969, 0.0125579, 0.0150983];

subplot(2,1,1);
plot(observe_frames, RMSE_translation_2, 'b');
ylabel('RMSE of translation');
xlabel('observe frames');
grid on;
hold on;
subplot(2,1,2);
plot(observe_frames, singular_value_ratio_2, 'y');
ylabel('singular value ratio');
xlabel('observe frames');
grid on;