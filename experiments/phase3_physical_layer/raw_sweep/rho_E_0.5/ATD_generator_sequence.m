%% ATD_generator_sequence.m
% Artificial Training Data Generator
% Sequence representation for LSTM

clearvars -except K L rho_u rho_E beta_k beta_E T T0 T_hat;

if ~exist('K','var'), K = 4; end
if ~exist('L','var'), L = 10; end
if ~exist('rho_u','var'), rho_u = 5; end
if ~exist('rho_E','var'), rho_E = 5; end
if ~exist('beta_k','var'), beta_k = 1; end
if ~exist('beta_E','var'), beta_E = 1; end
if ~exist('T','var'), T = 50; end
if ~exist('T_hat','var'), T_hat = 2000; end

sequence_rows = T_hat * 2;
ATD_sequence = zeros(sequence_rows, T + 1);
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

    ATD_sequence(row,1:T) = z_non;
    ATD_sequence(row,T+1) = 0;
    row = row + 1;

    ATD_sequence(row,1:T) = z_attack;
    ATD_sequence(row,T+1) = 1;
    row = row + 1;
end

fprintf('\nSequence Dataset\n');
fprintf('%d rows\n', size(ATD_sequence,1));
fprintf('%d columns\n', size(ATD_sequence,2));

sequence_table = array2table(ATD_sequence);

sequence_names = cell(1,T+1);
for t = 1:T
    sequence_names{t} = sprintf('Z_%d',t);
end
sequence_names{T+1} = 'LABEL';

sequence_table.Properties.VariableNames = sequence_names;

writetable(sequence_table, 'ATD_sequence.csv');

fprintf('\nSaved as ATD_sequence.csv\n');
disp('First Five Sequence Samples');
disp(ATD_sequence(1:min(5,size(ATD_sequence,1)),1:min(10,T+1)));
