function matches_after = RPFM_2sides(model,part,main_params,matches_gt)

    if ~isfield(main_params,'num_deltas'), num_deltas = 1000; else, num_deltas = main_params.num_deltas; end
    

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
%         est_rank = sum(part.evals - max(model.evals)<0);         
        est_rank = round(min( sum(calc_tri_areas(part))/sum(calc_tri_areas(model)), ...
            sum(calc_tri_areas(model))/sum(calc_tri_areas(part))) * main_params.num_eigen);
    else
        est_rank = main_params.rank
    end
    k = main_params.num_eigen;    
    x0(:,:,1) = eye(k); x0(:,:,2) = eye(k);   %X(:,:,1) -> model, and X(:,:,2) -> part
    C = blkdiag(eye(est_rank),zeros(k-est_rank)); % truncated diagonal
    lambda_m = diag(model.evals);
    lambda_p = diag(part.evals);
    W = 1 - diag(ones(k,1)); % off-diagonal mask
    mu = main_params.mu.dense;
    
    for reiterate = 1:main_params.num_reiterate+1 
        
        functions.fun_f = @(X)sum(sum( ( (X(:,:,1)'*lambda_m*X(:,:,1)) .*W).^2 + ( (X(:,:,2)'*lambda_p*X(:,:,2)) .*W).^2 ));
        functions.dfun_f = @(X)reshape([4*(lambda_m*X(:,:,1)*X(:,:,1)'*lambda_m*X(:,:,1) - (repmat(diag(X(:,:,1)'*lambda_m*X(:,:,1))',k,1)).*(lambda_m*X(:,:,1))),...
                            + 4*(lambda_p*X(:,:,2)*X(:,:,2)'*lambda_p*X(:,:,2) - (repmat(diag(X(:,:,2)'*lambda_p*X(:,:,2))',k,1)).*(lambda_p*X(:,:,2)))],k,k,2);
        
        % now the non-smooth function v (that's the fuction inside the l21 norm)
        functions.fun_v = @(X)C*(X(:,:,2)'*A - X(:,:,1)'*B);
        
        functions.fun_h = @(X,Z,U)0.5*sum( sum( ( -Z + U + C*(X(:,:,2)'*A - X(:,:,1)'*B)).^2 ) );        
        functions.dhdx = @(X,Z,U)reshape([B*B'*X(:,:,1)*C'*C - B*(C*X(:,:,2)'*A + U - Z)'*C ,...
                             A*A'*X(:,:,2)*C'*C + A*(-C*X(:,:,1)'*B + U - Z)'*C],k,k,2);
        
        params.lambda = mu;
        params.rho = mu;
        params.manifold = stiefelfactory(k, k, 2);
        params.is_plot = main_params.verbose;
        if ~isfield(main_params,'max_iter'), params.max_iter = 100; else,  params.max_iter  = main_params.max_iter; end;
        if ~isfield(main_params,'manopt_maxiter'), params.manopt_maxiter = 100; else,  params.manopt_maxiter  = main_params.manopt_maxiter; end;        
        if ~isfield(main_params,'icp_maxiter'), params.icp_maxiter = 0; else, params.icp_maxiter = main_params.icp_maxiter; end;
%         x0 = X_out
        figure,
        X_out = madmm_l21(x0,functions,params);
        
             
        %% show C_out
        A_ = C*X_out(:,:,2)'*A;
        B_ = C*X_out(:,:,1)'*B;
        C_out = (B_'\A_')';
        
        if main_params.verbose
            figure,imagesc(C_out);colorbar;title('After diagonalization');caxis([-1 1]);axis image
        end
        
        
        %% show indicator
        model.evecs_new = model.evecs*X_out(:,:,1)*C';
        part.evecs_new = part.evecs*X_out(:,:,2)*C';
        
        uu_new = part.evecs_new'*part.S*ones(numel(part.X),1);
        if main_params.verbose
            figure(998), showshape(model,model.evecs_new*uu_new,[35 20]);title('after diagonalization');caxis([-1 1]);lighting none
        end
        
        vv_new = model.evecs_new'*model.S*ones(numel(model.X),1);
        if main_params.verbose
            figure(999), showshape(part,part.evecs_new*vv_new,[35 20]);title('after diagonalization');caxis([-1 1]);lighting none
        end
                        
        model_ = reformat_shape(model);
        part_ = reformat_shape(part);        
        [~,matches_after] = run_icp(model_, part_, est_rank, C, params.icp_maxiter)
        
        X = part_.evecs' * part_.S;
        Y = model_.evecs' * model_.S;
        
        errors = sqrt(sum((C*X - Y(:,matches_after)).^2));
        [~,I] = sort(errors);
        I = I(1:round(0.9*numel(I)))';
        fps = fps_euclidean(part_.VERT(I,:),num_deltas,randi([1 numel(I)])); 
        fps = I(fps);
        FG = compute_indicator_functions({part_,model_}, [fps matches_after(fps)]', main_params.delta_radius);
        
        F = FG{1};
        G = FG{2};
        
        if main_params.verbose
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


end