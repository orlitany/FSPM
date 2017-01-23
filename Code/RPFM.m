function matches_after = RPFM(model,part,main_params,matches_gt)
    
    
    %% prepare eigenfunctions
    if ~isfield(model,'evecs'), [model.evecs,~,model.evals,model.S] = extract_eigen_functions_new(model,main_params.num_eigen);end
    if ~isfield(part,'evecs'),  [part.evecs,~,part.evals,part.S] = extract_eigen_functions_new(part,main_params.num_eigen); end
   
    %% prepare descriptors   
    if ~isfield(model,'shot'), model.shot = my_calc_shot(model, main_params); end
    if ~isfield(part,'shot'), part.shot = my_calc_shot(part, main_params); end    
    
    if exist('matches_gt'),        
        part_.TRIV = part.TRIV; part_.VERT = [part.X part.Y part.Z];
        model_.TRIV = model.TRIV; model_.VERT = [model.X model.Y model.Z];
        pts = [1:numel(part.X)]';
        FG = compute_indicator_functions({part_,model_}, [pts matches_gt(pts)]', main_params.delta_radius);
        
        part.shot = FG{1};
        model.shot = FG{2};

    end
    
    if main_params.verbose
        %% show shape
        figure, subplot(121); showshape(part); title('Part')
        subplot(122), showshape(model);title('Full shape')
    end    
    
    %% Solve for the best orthogonal transformation
    A = part.evecs'*part.S*part.shot;
    B = model.evecs'*model.S*model.shot;   
    if ~isfield(main_params,'rank'), 
        est_rank = sum(part.evals - max(model.evals)<0);  
    else
        est_rank = main_params.rank
    end 
    k = main_params.num_eigen;    
    x0 = eye(k);    
    C = blkdiag(eye(est_rank),zeros(k-est_rank)); % truncated diagonal
    lambda1 = diag(model.evals);
    W = 1 - diag(ones(k,1)); % off-diagonal mask
    mu = main_params.mu.dense;
    
    for reiterate = 1:main_params.num_reiterate+1 
        
        functions.fun_f = @(X)sum(sum( ( (X'*lambda1*X) .*W).^2 ));
        functions.dfun_f = @(X)4*(lambda1*X*X'*lambda1*X - (repmat(diag(X'*lambda1*X)',k,1)).*(lambda1*X));
        
        % now the non-smooth function v (that's the fuction inside the l21 norm)
        functions.fun_v = @(X)C*(A - X'*B);
        
        functions.fun_h = @(X,Z,U)0.5*sum( sum( ( -Z + U + C*(A - X'*B)).^2 ) );
        functions.dhdx = @(X,Z,U)((B*B')*X*C'*C - (B*(C*A + U - Z)')*C)
        
        params.lambda = mu;
        params.rho = mu;
        params.manifold = stiefelfactory(k, k);
        params.is_plot = main_params.verbose;
        
        if ~isfield(main_params,'max_iter'), params.max_iter = 100; else,  params.max_iter  = main_params.max_iter; end;
        if ~isfield(main_params,'manopt_maxiter'), params.manopt_maxiter = 100; else,  params.manopt_maxiter  = main_params.manopt_maxiter; end;                
        if ~isfield(main_params,'icp_maxiter'), params.icp_maxiter = 0; else, params.icp_maxiter = main_params.icp_maxiter; end;
        if ~isfield(main_params,'num_deltas'), params.num_deltas = 1000; else, params.num_deltas = main_params.num_deltas; end;
        
        if isfield(main_params,'is_track_time'), params.is_track_time = main_params.is_track_time; end
        figure,
        X_out = madmm_l21(x0,functions,params);
        if isfield(X_out,'t'), time_log = X_out.t; X_out = X_out.X; end;
        
             
        %% show C_out
        A_ = C*A;
        B_ = C*X_out'*B;
        C_out = (A_'\B_')';
        
        if main_params.verbose
            figure,imagesc(C_out);colorbar;title('After diagonalization');caxis([-1 1]);axis image            
        end
        
        
        %% show indicator
        model.evecs_new = model.evecs*X_out*C';
        part.evecs_new = part.evecs*C';
        
        uu_new = part.evecs_new'*part.S*ones(numel(part.X),1);
        if main_params.verbose
            figure(999), showshape(model,model.evecs_new*uu_new,[35 20]);title('after diagonalization');caxis([-1 1]);
        end
                        
        model_ = reformat_shape(model);
        part_ = reformat_shape(part);        
        [~,matches_after] = run_icp(model_, part_, est_rank, C, params.icp_maxiter)
       
        X = part_.evecs' * part_.S;
        Y = model_.evecs' * model_.S;
        
        errors = sqrt(sum((C*X - Y(:,matches_after)).^2));
        [~,I] = sort(errors);
        I = I(1:round(0.9*numel(I)))';
        fps = fps_euclidean(part_.VERT(I,:),params.num_deltas,randi([1 numel(I)])); 
        fps = I(fps);
        FG = compute_indicator_functions({part_,model_}, [fps matches_after(fps)]', main_params.delta_radius);
        
        F = FG{1};
        G = FG{2};
        
        if main_params.verbose
            ind = zeros(part_.n,1);ind(fps)=1;figure,showshape(part_,ind);caxis([0 1])
            ind = zeros(model_.n,1);ind(matches_after(fps))=1;figure,showshape(model_,ind);caxis([0 1])
        end
        
        colors = create_colormap(model_,model_);
        figure(2);subplot(1,3,1);colormap(colors);
        plot_scalar_map(model_,[1: size(model_.VERT,1)]');freeze_colors;title('Model');
                
        figure(2);subplot(1,3,3);colormap(colors(matches_after,:));
        plot_scalar_map(part_,[1: size(part_.VERT,1)]');freeze_colors;title('After slantization');
        
        %% Re-iterate with point-wise corr.
        A = part.evecs'*part.S*F;
        B = model.evecs'*model.S*G;
        mu = main_params.mu.sparse;
        x0 = X_out;
    end

if exist('time_log'), matches_after_.matches = matches_after; matches_after_.time_log = time_log; matches_after = matches_after_; end;
end