% This script runs the experiment for perfect isometry using *SHOT*
clear all;close all;
log_file = {};
%% Dependencies
addpath(genpath('./Code/'))
addpath(genpath('./Data/'))
addpath(genpath('./../Utils/'))
addpath(genpath('./../../3D_shapes_tools/'))
addpath(genpath('./../../manopt/'))
%% load shapes to match
log_file.model_filename = 'cat0';
log_file.part_filename = 'cat0_parts'
data_folder = './../../nonRigidPuzzle\Data\fromEma\4\';
load([data_folder log_file.part_filename]);
load(['./../../nonRigidPuzzle\Data\fromEma\' log_file.model_filename])
part = parts{1};

log_file.fullshape_idx = part.fullshape_idx;
log_file.part_evecs = part.evecs;
log_file.model_evecs = M.evecs;
%% show shape
figure, subplot(121); showshape(part); title('Part')
subplot(122), showshape(M);title('Full shape')

%% show C matrix
%- truncate after k eigen functions
k = 90;
log_file.k = k;

part.evecs = part.evecs(:,1:k);
part.evals = part.evals(1:k);
M.evecs = M.evecs(:,1:k);
M.evals = M.evals(1:k);

% using ground truth indices
part.shot = M.shot(part.fullshape_idx,:);

%- Construct C matrix using ground-truth correspondence and compare to A\B
nn = size(M.VERT,1);
mm = size(part.VERT,1);
P = zeros(nn,mm);
P(sub2ind([nn mm],part.fullshape_idx,[1:numel(part.fullshape_idx)]'))=1;

C_init = M.evecs'*M.S*P*part.evecs;
log_file.C_init = C_init;

A = part.evecs'*part.S*part.shot;
B = M.evecs'*M.S*M.shot;


%% Solve for the best orthogonal transformation
est_rank = sum(part.evals - max(M.evals)<0);
% est_rank = 90;
log_file.est_rank = est_rank;

problem.M = stiefelfactory(k, est_rank);
lambda1 = diag(M.evals);
mu = 1e-1;
% W = 1 - diag(ones(k,1)); % off-diagonal mask
W = 1 - diag(ones(est_rank,1));
rnk = est_rank;

% C = blkdiag(eye(rnk),zeros(k-rnk)); % truncated diagonal
C = [eye(rnk) zeros(rnk,k-rnk)];

problem.cost  =  @(X)mu*sum(sum( (C*A - X'*B).^2 )) + sum(sum( ( (X'*lambda1*X) .*W).^2 )) ; %+ sum(sum( (lambda2 .*W).^2 ))
problem.egrad = @(X)( 2*mu*(B*B'*X - B*(C*A)') + 4*(lambda1*X*X'*lambda1*X - (repmat(diag(X'*lambda1*X)',k,1)).*(lambda1*X)) );

checkgradient(problem);
options.maxiter = 10000;
% x0 = eye(k);
x0 = [eye(rnk);zeros(k-rnk,rnk)];
X_out = conjugategradient(problem,x0,options);

log_file.X_out = X_out;


%% show C_out
figure;subplot(1,2,1);imagesc(C_init);colorbar;title('Initial C');caxis([-1 1]);axis image
A_ = C*A;
B_ = C*X_out'*B;
log_file.C_least_sqaures = (A_'\B_')';

subplot(1,2,2);imagesc(log_file.C_least_sqaures);colorbar;title('After diagonalization');caxis([-1 1]);axis image
%% show the orthogonal matrices S and T
C_out = X_out*C;
log_file.C_out = C_out;
figure;imagesc(C_out);caxis([-1 1]);axis image;title('orthogonal matrix - model')

%% show indicator
figure(1);
uu = C_init*part.evecs'*part.S*ones(part.n,1);
subplot(1,2,1);showshape(M,M.evecs*uu,[35 20]);title('using ground-truth C');caxis([-1 1]);colormap(jet)

M.evecs_new = M.evecs*X_out*C';
part.evecs_new = part.evecs*C';

log_file.part_evecs_new = part.evecs_new;
log_file.model_evecs_new = M.evecs_new;
%     uu = Phi_new'*part{1}.M*ones(numel(part{1}.shape.X),1);
% uu_new = C_out*part.evecs_new'*part.S*ones(part.n,1);
indicator_vector = M.evecs_new*part.evecs_new'*part.S*ones(part.n,1);
figure(1);subplot(1,2,2);showshape(M,double(indicator_vector>0.5),[35 20]);title('after diagonalization');caxis([-1 1]);colormap(jet)
% figure, subplot(1,2,2);showshape(M,M.evecs_new(:,1),[35 20]);title('after diagonalization');caxis([-1 1]);colormap(jet)
log_file.indicator_vector = indicator_vector;
%% draw matches on shape
%- find NN in frequency domain: 
%-  original basis
[~,matches_before] = run_icp(M, part, est_rank, C_init, 0)
%-  new basis
M_ = M;M_.evecs = M.evecs_new;
part_ = part;part_.evecs = part.evecs_new;
% [~,matches_after] = run_icp(M_, part_, est_rank, C_out, 0)
[~,estimated_matches] = run_icp(M_, part_, est_rank, C, 0)

% draw
colors = create_colormap(M,M);
figure(2);subplot(1,3,1);colormap(colors);
plot_scalar_map(M,[1: size(M.VERT,1)]');freeze_colors;title('Model');
subplot(1,3,2);colormap(colors(matches_before,:));
plot_scalar_map(part,[1: size(part.VERT,1)]');freeze_colors;title('Before slantization');

figure(2);subplot(1,3,3);colormap(colors(estimated_matches,:));
plot_scalar_map(part,[1: size(part.VERT,1)]');freeze_colors;title('After slantization');


log_file.estimated_matches = estimated_matches;

% save('cat_RPFM', 'log_file')
