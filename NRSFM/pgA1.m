
function [X,M,S,t,S3,output] = pgA1( W, D, K, B, lambda, X0, options )
%function [X,M,S,t,S3,output] = pgA1( W, D, K, B, X0, options )
%
% CSF2: Matrix factorization for non-rigid structure from motion (no occlusion)
%
% By Paulo Gotardo
%
% Inputs:
%
% W is the observation matrix (here it is assumed complete; no missing data)
% D is the block diagonal matrix of camera projections (orthographic model)
% K is the factorization rank parameter (number of basis shapes)
% B is the DCT basis or KPCA basis obtained with a RIK
% X0 is the initial coefficient matrix
%
% Outputs:
%
% X is a compact representation of shape coefficients w.r.t. basis B
% M is the 2T-by-3K motion factor
% S is the 3K-by-n  factor with K basis shapes (each one is 3-by-n)
% t is the mean column vector of W (2D translations due to camera motion)
% S3 is 3T-by-n, the t^th triplet or rows has the recovered 3D shape for image t
%
pgCheckBuildKronmex()


% (1) Initialization: estimate t and make W zero mean
[T,n] = size(W); T=T/2;
I3 = eye(3);

if isfield(options,'A2');
    Wc = W;
else
    t = mean(W,2);
    Wc = W - repmat(t, 1, n);  %减去均值
end

d = size(B,2);                              % number of basis vectors in B
if ~exist('X0','var') || isempty(X0)
    X0 = eye(d,K);    %初始化为单位矩阵                      % deterministic initialization of X
end

Bnr = D * kronmex( B, I3 );
V   = pgVecAxI (d, K, 3);           % mapping matrix: vec(kron(X,I3)) = V vec(X)
I2T = speye(2*T); %稀疏单位矩阵 2行T列
P    = cell(K,1);                   % projections orthogonal to each Mk  % P是一个K行1列的cell 型矩阵
cols = 1:(9*d);                     % column indices for Jacobian terms
Jj = zeros(2*T, 9*d*K);             % Jacobian matrix
Ji = zeros(2*T, 9*d*K);             % Jacobian matrix

% (2) Optimization ------------------------------------------------------------
switch options.Method
    case 'MATLAB', method = @fminunc;
    case 'pgDN'  , method = @pgDampedGaussNewton;
    otherwise    , addpath ../SFM_utils/minFunc/; method = @minFunc;
end
options.OutputFcn = @pgOutputFunction;
[ vecX,fval,exitflag,output ] = method( @pgCostFunction, X0(:), options );

% (3) Finalization ------------------------------------------------------------
[X,C,M,S] = updateFactors( vecX );
S3 = kronmex( C, I3 ) * S;

% return % (done!)
%% -----------------------------------------------------------------------------
%更新M和S
%piM
    function [X,C,M,S,piM] = updateFactors( vecX )
        
        X = reshape( vecX, d,K );
        C = B * X;
        M = D * kronmex( C, I3 );
        [S,piM] = pgGetS( Wc, M );
    end
% -----------------------------------------------------------------------------
%% Evaluates function (f), gradient (g), and Hessian (H)

    function [f,g,H] = pgCostFunction( vecX)
        
        % (1) compute factors M, M+, S, R, and RMSE
        [X,C,M,S,piM] = updateFactors( vecX );
        R = Wc - M*S;                         % error matrix     (residues)
        
        % revised
        R_re = 0;   %R_re = zeros(2*T, n-1);
        c22 = 0;
        for re_i = 1:n-1
%             R_rei = Wc(:, re_i) - M*S(:, re_i);                         % error matrix     (residues)
            for re_j = re_i+1:n
%                 R_rej = Wc(:, re_j) - M*S(:, re_j);
                R_re =  R_re + (R(:, re_i) - R(:, re_j))'*(R(:, re_i) - R(:, re_j));    
                c22 = c22+1;
            end
        end
        % revised end
        
        % 2D fit error, f(M)
        c = 1 / numel(Wc);     % numel:矩阵元素的个数
        %   c2 = 1/(T*n*(n-1));
        c2 = 1/(2*T*c22);
        
        % revised
        % original code： f = (c/2) * ( R(:)'*R(:));
        % f = (c/2) * ( R(:)'*R(:)) + (c2/2)*lambda*(R_re(:)'*R_re(:));    %向量2范数，向量元素的平方和再开方
        f = (c/2) * ( R(:)'*R(:)) + (c2/2)*lambda*R_re;
        % revised end
        
        %f = sqrt(mean( R(:).^2 ));   %这一句没用到，和论文中有所不同
        
        
        if (nargout < 2), return, end         % all done
        g1 = zeros(d*K,1);     g2 = zeros(d*K,1);
        if (nargout > 2), H1 = zeros(d*K); H2 = zeros(d*K);end
        
        %K是标量
        % (2) Compute projection onto the orthogonal space of each triplet Mk
        P{K} = (I2T - M(:, 3*K-[2 1 0]) * piM{K});
        for k = (K-1):-1:1
            P{k} = P{k+1} * (I2T - M(:, 3*k-[2 1 0]) * piM{k});
        end
        for k = 1:K
            P{k} = P{k}*Bnr;
        end
        
        % (3) compute the gradient vector and optionaly the Hessian matrix
        for j = 1:n
            for k = 1:K
                sjk = S(3*k-[2 1 0], j);
                dc = (k-1)*9*d;
                %Jj(:,cols+dc) = kronmex( sjk' , P{k} );
                Jj(:,cols+dc) = [ sjk(1)*P{k} sjk(2)*P{k} sjk(3)*P{k} ];
            end
            JjV = Jj*V;
            g1 = g1 - (R(:,j)' * JjV)';    %求所有j个点的梯度
            
            if (nargout < 3), continue, end
            H1 = H1 + JjV'*JjV;    %求所有j个点的J矩阵
        end
        % Enforce symmetry on H
        H1 = (c/2) * (H1 + H1');   %令H是一个对称矩阵，求均值
        g1 = c * g1;   %求g的均值
        
        % revised
        for j = 1:n-1
            for i = j+1:n
%                 Ri = Wc(:, i) - M*S(:, i);                         % error matrix     (residues)
%                 Rj = Wc(:, j) - M*S(:, j);           
%                 Rji = Rj - Ri;
                Rji = R(:, j) - R(:, i);
                for k = 1:K
                    sjk = S(3*k-[2 1 0], j);
                    sik = S(3*k-[2 1 0], i);
                    dc = (k-1)*9*d;
                    %Jj(:,cols+dc) = kronmex( sjk' , P{k} );
                    Jj(:,cols+dc) = [ sjk(1)*P{k} sjk(2)*P{k} sjk(3)*P{k} ];
                    Ji(:,cols+dc) = [ sik(1)*P{k} sik(2)*P{k} sik(3)*P{k} ];
                end
                JijV = (Jj - Ji)*V;
                g2 = g2 - (Rji' * JijV)';    %求所有j个点的梯度
                
                if (nargout < 3), continue, end
                H2 = H2 + JijV'*JijV;    %求所有j个点的J矩阵
            end
        end
        % Enforce symmetry on H
        H2 = (c2/2) * (H2 + H2');   %令H是一个对称矩阵，求均值
        g2 = c2 * g2;   %求g的均值
        %revised end
        g = g1 + lambda*g2;
        H = H1 + lambda*H2;
        
    end

% ----------------------------------------------------------------------------
    function stop = pgOutputFunction ( vecX, optvals, state )
        
        persistent str
        
        stop = false;
        switch state
            case 'iter'
                % delete previously displayed line
                fprintf(repmat( '\b', 1, numel(str) ))
                rmse = sqrt( 2*optvals.fval );
                str = sprintf(' i = %-3d \t RMSE = %-15.10g', optvals.iteration, rmse);
                fprintf('%s',str)
                
            case 'interrupt'
                % probably no action here. Check conditions to see whether optimization should quit.
            case 'init'
                % setup for plots or guis
                str = [];
                fprintf('\n')
            case 'done'
                % cleanup of plots, guis, or final plot
                fprintf('\n')
            otherwise
                fprintf('\b.\n')           % display simple progress indicator
        end
    end
% ----------------------------------------------------------------------------
end                                                    % end of main function
% ----------------------------------------------------------------------------
% Computes S from Wc and M

function [S,piM] = pgGetS (Wc, M)

K = size(M,2) / 3;   % M: 2T*3K
n = size(Wc,2);   % Wc 2T*n
S = zeros(3*K,n);

piM = cell(K,1);
%M是每三列是一个组合, 对应形状基S是每三行
for k = 1:K, k3 = 3*k-[2 1 0];
    Mk = M(:,k3);
    piM{k} = pinv(Mk); %
    Sk = piM{k} * Wc;
    Wc = Wc - Mk * Sk;
    S(k3,:) = Sk;
end

end
% ----------------------------------------------------------------------------
% Magnus&Neudecker (execise pg48):
% vec(kron(Ih,A)) = kron(H,Im) vec(A) = Ch vec(A), where A is m-by-n

function K = pgVecAxI (m,n, i)

I = speye(i);  %稀疏单位矩阵i*i
G = kron(pgKmn(i,m), I) * kron(speye(m), I(:)); %kron 计算两个矩阵的Kronecker积，kron(A,B)就是矩阵A中的每个元素都乘以矩阵B
K = kron(speye(n), G);

end
% ----------------------------------------------------------------------------
% Commutation matrix for m-by-n A: vec(A') = K * vec(A);

function Kmn = pgKmn (m, n)

na = m*n;
ind = 1:na;
pos = reshape(ind, [ m n ])';
%pos是n*m索引矩阵，其中元素按行：1, 2, 3, 4, 5 ... m*nm, pos(:)按列将pos拉成一列
Kmn = sparse( ind, pos(:), ones(na,1), na, na ); %Kmn中只有规定的部分元素为1，其它部分为0
% Kmn = zeros(na,na);
% for row = 1:na
%     Kmn(row, pos(row)) = 1;
% end
end
% ----------------------------------------------------------------------------
