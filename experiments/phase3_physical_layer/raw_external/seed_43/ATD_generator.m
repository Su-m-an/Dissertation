%% ATD_generator.m
% Artificial Training Data Generator
% Feature representation: MEAN, RATIO, LABEL

clearvars -except K L rho_u rho_E beta_k beta_E T T0 T_hat;

if ~exist('K','var'), K = 4; end
if ~exist('L','var'), L = 10; end
if ~exist('rho_u','var'), rho_u = 5; end
if ~exist('rho_E','var'), rho_E = 5; end
if ~exist('beta_k','var'), beta_k = 1; end
if ~exist('beta_E','var'), beta_E = 1; end
if ~exist('T','var'), T = 50; end
if ~exist('T0','var'), T0 = 5; end
if ~exist('T_hat','var'), T_hat = 2000; end

feature_rows = T_hat * 2 * (T - T0 + 1);
ATD_features = zeros(feature_rows, 3);
row = 1;

for i = 1:T_hat

    gk = sqrt(beta_k/2) .* (randn(K,T) + 1i*randn(K,T));
    gE = sqrt(beta_E/2) .* (randn(1,T) + 1i*randn(1,T));
    noise = sqrt(0.5) .* (randn(L,T) + 1i*randn(L,T));

    P = zeros(L,K);
    for k = 1:K
        P(k,k) = 1;
    end

    k_target = 1;
    p_k = P(:,k_target);
    p_E = p_k;

    noise_projected = p_k' * noise;
    noise_power = abs(noise_projected).^2;

    z_non = zeros(1,T);
    z_attack = zeros(1,T);

    for t = 1:T

        y_non = sqrt(L*rho_u) * (p_k * gk(k_target,t)) + noise(:,t);

        y_attack = sqrt(L*rho_u) * (p_k * gk(k_target,t)) + ...
                   sqrt(L*rho_E) * (p_E * gE(t)) + noise(:,t);

        y_non_proj = p_k' * y_non;
        y_attack_proj = p_k' * y_attack;

        z_non(t) = abs(y_non_proj)^2;
        z_attack(t) = abs(y_attack_proj)^2;
    end

    for tt = T0:T

        mean_non = mean(z_non(1:tt));
        ratio_non = (sum(z_non(1:tt)) - sum(noise_power(1:tt))) ...
                    / sum(noise_power(1:tt));

        ATD_features(row,:) = [mean_non, ratio_non, 0];
        row = row + 1;

        mean_attack = mean(z_attack(1:tt));
        ratio_attack = (sum(z_attack(1:tt)) - sum(noise_power(1:tt))) ...
                       / sum(noise_power(1:tt));

        ATD_features(row,:) = [mean_attack, ratio_attack, 1];
        row = row + 1;
    end
end

fprintf('\nFeature Dataset\n');
fprintf('%d rows\n', size(ATD_features,1));
fprintf('%d columns\n', size(ATD_features,2));

feature_table = array2table(ATD_features, ...
    'VariableNames', {'MEAN','RATIO','LABEL'});

writetable(feature_table, 'ATD_features.csv');

fprintf('\nSaved as ATD_features.csv\n');
disp('First Five Feature Samples');
disp(ATD_features(1:5,:));
