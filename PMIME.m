function [RM,ecC] = PMIME(allM,Lmax,T,nnei,A,showtxt)
% function [RM,ecC] = PMIME(allM,Lmax,T,nnei,A,showtxt)
% PMIME (Partial Mutual Information on Mixed Embedding)
% computes the measure R_{X->Y|Z} for all combinations of X and Y time
% series from the multivariate time series given in matrix 'allM', of size
% N x K, where Z contains the rest K-2 time series. 
% The components of X,Y, and Z, are found from a mixed embedding aiming at
% explaining Y. The mixed embedding is formed by using the progressive 
% embedding algorithm based on conditional mutual information (CMI). 
% CMI is estimated by the method of nearest neighbors (Kraskov's method). 
% The function is the same as PMIMEsig.m but defines the stopping criterion
% differently, using a fixed rather than adjusted threshold. Specifically,
% the algorithm terminates if the contribution of the selected lagged
% variable in explaining the future response state is small enough, as
% compared to a threshold 'A'. Concretely, the algorithm terminates if 
%        I(x^F; w| wemb) / I(x^F; w,wemb) <= A
% where I(x^F; w| wemb) is the CMI of the selected lagged variable w and 
% the future response state x^F given the current mixed embedding vector, 
% and I(x^F; w,wemb) is the MI between x^F and the augmented mixed
% embedding vector [wemb w].
% We experienced that in rare cases the termination condition is not 
% satisfied and the algorithm does not terminate. Therefore we included a 
% second condition for termination of the algorithm when the ratio 
% I(x^F; w| wemb) / I(x^F; w,wemb) increases in the last two embedding
% cycles. 
% The derived R measure indicates the information flow of time series X to
% time series Y conditioned on the rest time series in Z. The measure
% values are stored in a K x K matrix 'RM' and given to the output, where
% the value at position (i,j) indicates the effect from i to j (row to
% col), and the (i,i) components are zero.
% INPUTS
% - allM : the N x K matrix of the K time series of length N.
% - Lmax : the maximum delay to search for X and Y components for the mixed 
%          embedding vector [default is 5].
% - T    : T steps ahead that the mixed embedding vector has to explain.
%          Note that if T>1 the future vector is of length T and contains
%          the samples at times t+1,..,t+T [dafault is 1]. 
% - nnei : number of nearest neighbors for density estimation [default is 5]
% - A    : the threshold for the ratio of CMI over MI of the lagged variables
%          for the termination criterion.
% - showtxt : if 0 or negative do not print out anything, 
%             if 1 print out the response variable index at each run, 
%             if 2 or larger print also info for each embedding cycle [default is 1].
% OUTPUTS
% - RM   : A K x K matrix containing the R values computed by PMIME using
%          the surrogates for setting the stopping criterion. 
% - ecC  : cell array of K components, where each component is a matrix of 
%          size E x 5, and E is the number of embedding cycles. For each 
%          embedding cycle the following 5 results are stored:
%          1. variable index, 2. lag index, 3. CMI of the selected lagged
%          variable w and the future response state x^F given the current 
%          mixed embedding vector, I(x^F; w| wemb). 4. MI between x^F and 
%          the augmented mixed embedding vector [wemb w], I(x^F; w,wemb).
%          5. The ration of 3. and 4.: I(x^F; w| wemb)/I(x^F; w,wemb)  
%
%     Copyright (C) 2015 Dimitris Kugiumtzis
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
%=========================================================================
% Reference : D. Kugiumtzis, "Direct coupling information measure from 
%             non-uniform embedding", Physical Review E, Vol 87, 062918, 
%             2013
%             I. Vlachos, D. Kugiumtzis, "Non-uniform state space 
%             reconstruction and coupling detection", Physical Review E, 
%             Vol 82, 016207, 2010
% Link      : http://users.auth.gr/dkugiu/
%========================================================================= 
maxcomps = 20; % A safeguard, to make sure that the algorithm does not make  �?大嵌入周期为20
               % more than "maxcomps" embedding cycles. 
if nargin==5
    showtxt = 1;
elseif nargin == 4
    showtxt = 1;
    A = 0.03;
elseif nargin == 3
    showtxt = 1;
    A = 0.03;
    nnei = 5;
elseif nargin == 2
    showtxt = 1;
    A = 0.03;
    nnei = 5;
    T = 1;
elseif nargin == 1
    showtxt = 1;
    A = 0.03;
    nnei = 5;
    T = 1;
    Lmax = 5;
end
if isempty(A), A = 0.03; end
if isempty(nnei), nnei = 5; end
if isempty(T), T = 1; end
if isempty(Lmax), Lmax = 5; end
[N,K] = size(allM);         % 获取矩阵的行数和列数
wV = Lmax*ones(K,1);        % K*1的列向量wv
%% Standardization of the input matrix columnwise in [0,1]. 在[0,1]中�?�列标准化输入矩阵�??
minallV=min(allM);          % min(A)将A的每�?列作为一个向量，返回�?行向量包含了每一列的�?小元素�?�返回了每个序列的最小�??
rang=kron((1./range(allM)),ones(N,1));
allM=(allM-kron(minallV,ones(N,1))).*rang;
%% Build up the lag matrix from all variables  从所有变量中构建滞后矩阵
alllagM = NaN(N,sum(wV)); % lag matrix of all variables  �?有变量的滞后矩阵
indlagM = NaN(K,2); % Start and end of columns of each variable in lag matrix  滞后矩阵中每个变量的列的�?始和结束，是alllagM的索�?
count = 0;
for iK=1:K
    indlagM(iK,:) = [count+1 count+wV(iK)];  % 对矩阵indlagM的每�?行进行赋值，第一次[0+1�?0+Lmax]
    alllagM(:,indlagM(iK,1))=allM(:,iK); % lag=0    
    for ilag=1:wV(iK)-1  % lag=1,...,Lmax-1
        alllagM((ilag+1):end,indlagM(iK,1)+ilag)=allM(1:(end-ilag),iK);
    end
    count = count+wV(iK);
end
alllagM = alllagM(Lmax:end-T,:);  %截取了一部分的矩�?
[N1,alllags] = size(alllagM);   %获取延迟矩阵的行数和列数
%% Find mixed embedding and R measure for purpose: from (X,Y,Z) -> Y   进行嵌入并计算指�? R
RM = zeros(K,K);  
ecC = cell(K,1);
psinnei = psi(nnei); % Computed once here, to be called in several times   psinnei双伽马函数数�?
psiN1 = psi(N1); % Computed once here, to be called in several times
for iK=1:K
    if showtxt==1
        fprintf('%d..',iK);
    elseif showtxt>=2
        fprintf('Response variable index=%d.. \n',iK);
        fprintf('EmbeddingCycle  Variable  Lag  I(x^F;w|wemb)  I(x^F;w,wemb)  I(x^F;w|wemb)/I(x^F;w,wemb) \n');
    end
    Xtemp = NaN(N,T);
    for iT=1:T
        Xtemp(1:(end-iT),iT)=allM((1+iT):end,iK);
    end
    xFM = Xtemp(Lmax:end-T,:); % The future vector of response
    % First embedding cycle: max I(y^T, w), over all candidates w  第一次嵌入在�?有�?��?�的w中求max I(y^T, w)
    miV=NaN*ones(alllags,1);
    for i1=1:alllags
        % Compute the mutual information of future response and each one of  使用KNN估计来计算未来响应变量和每个候�?�滞后集合元素的MI
        % the cadidate lags using the nearest neighbor estimate
        xnowM = [xFM alllagM(:,i1)];  %合并矩阵，以第一次为例将xFM和alllagM中第�?列合�? ，xnowM每一行在空间里算�?个点
        [~, distsM] = annMaxquery(xnowM', xnowM', nnei+1); %生成k近邻矩阵和距离矩�?
        maxdistV=distsM(end,:)';
        nyFV = nneighforgivenr(xFM,maxdistV-ones(N1,1)*10^(-10));
        nwcandV = nneighforgivenr(alllagM(:,i1),maxdistV-ones(N1,1)*10^(-10));
        psibothM=psi([nyFV nwcandV]);
        miV(i1)=psinnei + psiN1 - mean(sum(psibothM,2));
    end
    [~,iembV]=max(miV);
    xembM=alllagM(:,iembV);
    % add the selected lag variable in the first embedding cycle and
    % show it
    varind = ceil(iembV/Lmax); % the variable
    lagind = mod(iembV,Lmax);  % the lag of the variable
    if lagind==0
        lagind = Lmax;
    end
    ecC{iK}= [varind lagind miV(iembV) NaN NaN]; % For the first component
    if showtxt>=2
        fprintf('%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n',...
            size(ecC{iK},1),ecC{iK}(end,1),ecC{iK}(end,2),...
            ecC{iK}(end,3),ecC{iK}(end,4),ecC{iK}(end,5));
    end
    % End of first embedding cycle, the first lagged variale is found  第一个嵌入周期结束时，找到第�?个滞后变�?
    terminator=0; % Flag for terminating the embedding cycles 用于终止嵌入周期的标�?
    maxcomps = min(size(alllagM,2),maxcomps); % To avoid large embedding  
    % Run iteratively, for each embedding cycle select w from max I(y^; w | wemb)迭代运行对于每个嵌入循环，从max中�?�择w�?
    while (terminator==0 && size(xembM,2)<maxcomps)
        activeV = setdiff((1:alllags),iembV); % The indexed of the candidates  候�?�集索引
        cmiV=NaN*ones(alllags,1); % I(y^; w | wemb)   条件互信�?
        miwV=NaN*ones(alllags,1); % I(y^; w, wemb)    互信�?
        for i1=activeV
            % For each candidate lag w compute I(y^; w | wemb) and I(y^; w, wemb)
            xallnowM = [xFM alllagM(:,i1) xembM];
            [~, distsM] = annMaxquery(xallnowM', xallnowM', nnei+1);
            maxdistV=distsM(end,:)';
            nwV=nneighforgivenr(xembM,maxdistV-ones(N1,1)*10^(-10));
            nwcandV=nneighforgivenr([alllagM(:,i1) xembM],maxdistV-ones(N1,1)*10^(-10));
            nyFwV=nneighforgivenr([xFM xembM],maxdistV-ones(N1,1)*10^(-10));            
            psinowM = NaN*ones(N1,3);
            psinowM(:,1) = psi(nyFwV);
            psinowM(:,2) = psi(nwcandV);
            psinowM(:,3) = -psi(nwV);
            cmiV(i1) = psinnei - mean(sum(psinowM,2));
            nyFV = nneighforgivenr(xFM,maxdistV-ones(N1,1)*10^(-10));
            psinowM = [psi(nyFV) psinowM(:,2)];
            miwV(i1) = psinnei + psiN1 - mean(sum(psinowM,2));
        end
        [~,ind]=max(cmiV); % ind: index of the selected lagged variable �?选之后变量的索引
        xVnext=alllagM(:,ind); 
        varind = ceil(ind/Lmax); % the variable  变量
        lagind = mod(ind,Lmax);  % the lag of the variable  变量的嵌�?
        if lagind==0
            lagind = Lmax;
        end
        % The corrected termination criterion  修正的终止标�?
        ecC{iK}= [ecC{iK}; [varind lagind cmiV(ind) miwV(ind) cmiV(ind)/miwV(ind)]];
        if length(iembV)==1 
            % This is the second embedding cycle to be tested, use only
            % the criterion for the contribution of the selected lagged
            % variable  这是要测试的第二个嵌入周期，仅使用所选滞后变量贡献的标准
            if showtxt>=2
                fprintf('%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n',...
                    size(ecC{iK},1),ecC{iK}(end,1),ecC{iK}(end,2),...
                    ecC{iK}(end,3),ecC{iK}(end,4),ecC{iK}(end,5));
            end
            if ecC{iK}(end,5)>A
                xembM=[xembM xVnext];
                iembV=[iembV ind]; % The index of the subsequent component is added 添加后续索引
            else
                terminator=1;
            end
        else
            if showtxt>=2
                fprintf('%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n',...
                    size(ecC{iK},1),ecC{iK}(end,1),ecC{iK}(end,2),...
                    ecC{iK}(end,3),ecC{iK}(end,4),ecC{iK}(end,5));
            end
            if length(iembV)==2
                % This is the third embedding cycle to be tested, use only
                % the criterion for the contribution of the selected lagged
                % variable  这是要测试的第三个嵌入周期，仅使用所选滞后变量贡献的标准
                if ecC{iK}(end,5)>A
                    xembM=[xembM xVnext];
                    iembV=[iembV ind]; % The index of the subsequent component is added
                else
                    terminator=1;
                end
            else
                % This is the fourth or larger embedding cycle to be tested
                % and terminate if B=I(y^; w | wemb) / I(y^; w, wemb) < A
                % and B(j)>B(j-1)>B(j-2) at each embedding cycle j.
                if ecC{iK}(end,5)>A && (ecC{iK}(end,5)<ecC{iK}(end-1,5) ...
                        || ecC{iK}(end-1,5)<ecC{iK}(end-2,5))
                    xembM=[xembM xVnext];
                    iembV=[iembV ind]; % The index of the subsequent component is added
                else
                    terminator=1;
                end
            end
        end % if iembV
    end % while not terminate
    %disp(iembV);
    %disp(xembM);
    % Identify the lags of each variable in the embedding vector, if not
    % empty, and compute the R measure for each driving variable.  确定嵌入向量中每个变量的滞后（如果不是空的话），并计算每个驱动变量的R度量�?
    if ~isempty(iembV) && ~isempty(find(iembV<indlagM(iK,1) | iembV>indlagM(iK,2),1))
        % Find the lags of the variables  找出变量的滞�?
        %disp(iembV);
        xformM = NaN(length(iembV),2);
        xformM(:,1) = ceil(iembV/Lmax); % The variable indices
        xformM(:,2) = mod(iembV,Lmax); % The lag indices for each variable
        xformM(xformM(:,2)==0,2)=Lmax;
        % Make computations only for the active variables, which are the 
        % variables included in the mixed embedding vector. 仅对活动变量进行计算，这些变量是包含在混合嵌入向量中的变量�??
        activeV = unique(xformM(:,1)); 
       
        % Store the lags of the response and remove it from the active
        % variable list  存储响应的滞后并将其从活动变量列表中删除
        if ~isempty(intersect(activeV,iK))
            inowV = find(xformM(:,1)==iK);
            xrespM = xembM(:,inowV);
            activeV = setdiff(activeV,iK);
        else
            xrespM = []; % This is the case where the response is not 这是在混合嵌入向量中未表示响应的情况
                         % represented in the mixed embedding vector 
        end
        KK = length(activeV); 
        %disp(activeV);
        indKKM = NaN(KK,2); % Start and end in xembM of the active variables 以活动变量的xembM�?始和结束
        iordembV = NaN(length(iembV),1); % the index for reordering the lag 
                        % matrix to set together lags of the same variable
        count = 0;
        for iKK=1:KK
            inowV = find(xformM(:,1)==activeV(iKK));
            indKKM(iKK,:)=[count+1 count+length(inowV)];
            iordembV(indKKM(iKK,1):indKKM(iKK,2))=inowV;
            count = count+length(inowV);
        end
        iordembV = iordembV(1:indKKM(KK,2));
        % The total embedding vector ordered with respect to the active
        % variables and their lags, except from the response 
        xembM = xembM(:,iordembV);
        % Compute the entropy for the largest state space, containing the
        % embedding vector and the future response vector. This is done
        % once for all active variables, to be used in the computation of R.
        if isempty(xrespM)
            xpastM = xembM;
        else
            xpastM = [xrespM xembM];
        end
        [~,dists] = annMaxquery([xFM xpastM]',[xFM xpastM]',nnei+1);
        maxdistV=dists(end,:)';
        nyFV = nneighforgivenr(xFM,maxdistV-ones(N1,1)*10^(-10));
        nwV = nneighforgivenr(xpastM,maxdistV-ones(N1,1)*10^(-10));
        psi0V = psi([nyFV nwV]);
        psinnei = psi(nnei);
        IyFw = psinnei+psi(N1)-mean(sum(psi0V,2)); % I(y^T; w)
        %disp(IyFw);
        % For each active (driving) variable build the arguments in 
        % I(y^T; w^X | w^Y w^Z) and then compute it. Note that w^X is not 
        % needed to be specified because it is used only to form 
        % w=[w^X w^Y w^Z], which was done above one for all active variables.
        for iKK=1:KK
            indnowV = (indKKM(iKK,1):indKKM(iKK,2))';
            irestV = setdiff((1:size(xembM,2)),indnowV);
            % Construct the conditioning embedding vector [w^Y w^Z],
            % considering the cases one of the two components is empty.
            if isempty(irestV) && isempty(xrespM)
                xcondM = [];
            elseif isempty(irestV) && ~isempty(xrespM)
                xcondM = xrespM;
            elseif ~isempty(irestV) && isempty(xrespM)
                xcondM = xembM(:,irestV);
            else
                xcondM = [xrespM xembM(:,irestV)];
            end
            % Compute I(y^T; w^X | w^Y w^Z) 
            %disp(size(xcondM));
            %disp(size(xpastM));
            if isempty(xcondM)
                IyFwXcond = IyFw;
            else
                nxFcondV = nneighforgivenr([xFM xcondM],maxdistV-ones(N1,1)*10^(-10));
                ncondV = nneighforgivenr(xcondM,maxdistV-ones(N1,1)*10^(-10));
                psinowV = [psi(nxFcondV) psi0V(:,2) -1*psi(ncondV)];
                IyFwXcond = psinnei - mean(sum(psinowV,2));
            end
            RM(activeV(iKK),iK) = IyFwXcond / IyFw;
        end % for all active variables 
    end % if not empty embedding vector
    %disp(ecC{iK});
    if ~isempty(ecC{iK})
        ecC{iK}(end,:)=[]; % Upon termination delete tha last selected component.
    end
end % for all variables 
%disp(RM);
if showtxt>0
    fprintf('\n');
end

function npV=nneighforgivenr(xM,rV)
npV = annMaxRvaryquery(xM', xM',rV, 1, 'search_sch', 'fr', 'radius', sqrt(1));
npV=double(npV);
npV(npV==0)=1;

