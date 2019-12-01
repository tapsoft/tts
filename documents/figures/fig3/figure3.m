data1 = load('characters_k_steps_loss.csv');
loss1 = data1(:,2);
steps1 = 1:40;

data2 = load('syllables_k_steps_loss.csv');
loss2 = data2(:,2);
disp(data2);
steps2 = 11:32;

figure; hold on;
plot(steps1, loss1);
plot(steps2, loss2);

xlim([0, 40]);
ylim([-0.1, 1.1]);
yticks(0:0.2:1);