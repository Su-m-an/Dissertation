%% generate_dataset.m
% Generate Artificial Training Data (ATD)
% for Active Eavesdropping Detection

clear;
clc;
close all;

fprintf('=============================================\n');
fprintf(' Active Eavesdropping Detection - ATD\n');
fprintf('=============================================\n\n');

%% Parameters
K = 4;
L = 10;
rho_u = 5;
rho_E = 5;
beta_k = 1;
beta_E = 1;
T = 50;
T0 = 5;
T_hat = 2000;

%% Generate feature-based ATD
fprintf('Generating feature-based ATD...\n');
ATD_generator;
fprintf('\nFeature-based ATD generation completed.\n');

%% Generate sequence-based ATD
fprintf('\nGenerating sequence-based ATD...\n');
ATD_generator_sequence;
fprintf('\nSequence-based ATD generation completed.\n');

fprintf('\n=============================================\n');
fprintf(' Dataset generation completed\n');
fprintf('=============================================\n');
