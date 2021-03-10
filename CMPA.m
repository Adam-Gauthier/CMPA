clear all; close all; clc;
Is= 0.01e-12;
Ib=0.1e-12;
Vb=1.3;
Gp=0.1;

diode = @(V) Is*(exp(1.2*V/.025)-1)+Gp*V-Ib*(exp(-1.2*(V+Vb)/0.025)-1);

V = linspace(-1.95,0.7,200);

I= diode(V);

I_rand = [];
for curr =I
    r = 0.8 + (1.2-0.8) .* rand(1,1);
    I_rand=[I_rand curr*r];
end
%polynomial
P4 = polyfit(V,I,4);
P4_r =polyfit(V,I_rand,4);
P8 = polyfit(V,I,8);
P8_r =polyfit(V,I_rand,8);
Y4 = polyval(P4,V);
Y4_r = polyval(P4_r,V);
Y8 = polyval(P8,V);
Y8_r = polyval(P8_r,V);
figure(1);

subplot(1,2,1);
hold on;
plot(V,I);
plot(V,I_rand);
plot(V,Y4);
plot(V,Y8);
legend('I','I_r','Fit4','Fit8');
title('plot command');
xlabel('Voltage');
ylabel('Current')
hold off;

subplot(1,2,2);
semilogy(V,abs(I));
hold on;
semilogy(V,abs(I_rand));
semilogy(V,abs(Y4_r));
semilogy(V,abs(Y8_r));
legend('I','I_r','Fit4','Fit8');
title('semilogy command');
xlabel('Voltage');
ylabel('Current')
hold off;

%fit command
fo_4 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff_4 = fit(V,I,fo_4);
If_4 = ff_4(V);
fo_3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff_3 = fit(V,I,fo_3);
If_3 = ff_3(V);
fo_2 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff_2 = fit(V,I,fo_2);
If_2 = ff_2(V);

figure(2);
hold on;
plot(V,I);
plot(V,I_rand);
plot(V,If_4);
plot(V,If_3);
plot(V,If_2);
legend('I','I_r','4 params','3 params','2 params');
title('fit command');
xlabel('Voltage');
ylabel('Current')
hold off;

%Neural net
inputs = V;
targets = I;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets); 
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;


