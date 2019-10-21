%
% Demo A1: NRSFM with Rotation Invariant Kernels
%
% -----------------------------------------------------------------------------

clear all
strings = {'walking','face1','stretch','dance','pickup','yoga'};
for i = 1:size(strings,2)
    % Datasets:
    NUM_DATASET = i;
    
    % Load data (original 3D shapes and observation matrix with 2D points)
    [S0,W,T,n] = pgLoadDataNRSFM( strings{NUM_DATASET} );
    
    DataStringName = strings{NUM_DATASET};
    
    %% -----------------------------------------------------------------------------
    % Model parameters for each dataset:
    Ks  = {  5 ,  5 ,  8 ,  7  , 3, 7};   % number of basis shapes (rank parameter)
    ds1 = { 0.2, 0.3, 0.2, 0.2, 0.2, 0.2 };   % number of vectors (%T) in KPCA basis (2D RIK)
    ds2 = { 0.1, 0.3, 0.2, 0.1 , 0.2, 0.2};   % (if using aSFM kernel)
    
    kernels = {'RIK','aSFM'};
    selK = kernels{1};
    
    K = Ks{ NUM_DATASET };
    switch selK
        case  'RIK', d = ceil( ds1{ NUM_DATASET } * T );
        case 'aSFM', d = ceil( ds2{ NUM_DATASET } * T );
    end
    
    % Optimization parameters:
    opts = optimset( optimset('fminunc'), 'Display','iter', 'MaxIter', 300, ...
        'TolFun', 1e-12, 'TolX', eps); opts.Method = 'pgDN';
    %% -----------------------------------------------------------------------------
    % NRSFM OPTIMIZATION:
    
    % Estimate block-diagonal rotation matrix D
    [D,Rs] = pgComputeD( W );           % Rs has the rotations in the diagonal of D
    
    %% Estimate KPCA basis B
    [B,Ve,s2] = pgComputeBasisKPCA( W, d, 0.99, selK );
    
    num_lambda = 0;   %revised
    for lambda = 0.0001:0.0001:0.01  %revised
        num_lambda = num_lambda+1;  %revised
        %ÔËÓÃ¾ØÕó·Ö½âÓÅ»¯X
        % Factorization W = M*S, with M = D*kron(BX,I): solve for X
        % revised
        [X,M,S,t,S3] = pgA1( W, D, K, B, lambda, [], opts );
        %original code
        %  [X,M,S,t,S3] = pgA1( W, D, K, B, [], opts );
        %revised end 
        
        % -----------------------------------------------------------------------------
        % Compare to ground truth and display result
        
        % nfd: add
        switch DataStringName
            case  {'pickup',  'yoga', 'drink', 'stretch'}
                args = {};
                
            case {'walking', 'face1', 'face2', 'jaws', 'dance'}
                args = { Rs };
        end
        
        % nfd:delete
        %if strcmpi(strings{NUM_DATASET}, 'stretch'), args = {}; else args = { Rs }; end
        
        [err3D, S0R, S3R, errt] = pgCompare3DShapes( S0, S3, args{:} );
         
        %         fprintf('(Dataset %d, %s), err3D (std) = %5.4f (%5.4f), (K = %d, d = %d)\n', ...
        %             NUM_DATASET, strings{NUM_DATASET}, err3D, std(errt(:)), K, d )
        fid = fopen('results.txt', 'a');  
        fprintf(fid, '(Dataset %d, %s), err3D (std) = %5.4f (%5.4f), (K = %d, d = %d, lambda = %5.4f)\n', ...
            NUM_DATASET, strings{NUM_DATASET}, err3D, std(errt(:)), K, d, lambda)
        fprintf('\n'); 
        fclose(fid);
    end  %revised
    %%pgShowShapes3D(S0R, S3R);
end
% -----------------------------------------------------------------------------
