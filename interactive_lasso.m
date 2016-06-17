function [Ws,Theta] = interactive_lasso(XYZ,sizes,alpha,lambda,linear_not_log,l1_not_l21)
% both linear and logistic function
rho = 1;
numX = sizes.numX;
numU = sizes.numU;
numZ = sizes.numZ;
numChunks = size(XYZ,2);
W = cell(numChunks,1);
for i=1:numChunks
    W{i,1} = zeros(numX+1,(numU+1),(numZ+1));
end
numCoefs = 10;
Theta = cell(numChunks,numCoefs);%zeros(numX+1,(numU+1)*(numZ+1)*numCoefs);
Gamma = cell(numChunks,numCoefs);%zeros(numX+1,(numU+1)*(numZ+1)*numCoefs);
for i=1:numCoefs
    for j=1:numChunks
        Theta{j,i}=zeros(numX+1,numU+1,numZ+1);
        Gamma{j,i}=zeros(numX+1,numU+1,numZ+1);
    end
end
eps_pri = 1e-2;
eps_dual = 1e-2;
MAX_ITER = 1000;
% numObservations = numDates*numCities;
numVariables = (numX+1)*(numU+1)*(numZ+1);
ls_dict = containers.Map;
% lambda = 0.01;
% alpha = 0.9;
lambdas = zeros(1,numCoefs);
lambdas(1,1)=lambda*sqrt(numX);lambdas(1,2)=lambda*sqrt(numU);lambdas(1,3)=lambda*sqrt(numZ);
lambdas(1,4)=lambda*sqrt(numX)*numZ;lambdas(1,5)=lambda*sqrt(numU)*numZ;
lambdas(1,6)=lambda*sqrt(numX)*numU;lambdas(1,7)=lambda*sqrt(numZ)*numU;
lambdas(1,8)=lambda*sqrt(numU)*numX;lambdas(1,9)=lambda*sqrt(numZ)*numX;
lambdas(1,1:9) = lambdas(1,1:9)*(1-alpha);
lambdas(1,10)=lambda*alpha;
% [Y,XZ_T_XZ,XZ_T_YY,w,u,s,v] = deal(XYZ{:});

tic;
for iter = 1:MAX_ITER
    for i=1:numChunks
        W{i,1}=update_B(Theta(i,:),Gamma(i,:),rho,linear_not_log,XYZ{1,i},ls_dict,numCoefs,numX,numU,numZ);
    end
    Theta_old = Theta;
    inds = {
            {[1,numX+1],[2,numU+1],[2,numZ+1]},   {[2,numX+1],[1,numU+1],[2,numZ+1]},	{[2,numX+1],[2,numU+1],[1,numZ+1]},...
            {[1,numX+1],[2,numU+1],[1,1]},      {[2,numX+1],[1,numU+1],[1,1]},...
            {[1,numX+1],[1,1],[2,numZ+1]},      {[2,numX+1],[1,1],[1,numZ+1]},...
            {[1,1],[1,numU+1],[2,numZ+1]},      {[1,1],[2,numU+1],[1,numZ+1]}
            };
    for i=1:size(inds,2)
        cur_ind = inds{1,i};
        cur_mats = cell(numChunks,1);
        for j=1:numChunks
            cur_mats{j,1} = W{j,1}+Gamma{j,i}/rho;
        end
        Theta(:,i) = regularizer(cur_mats,XYZ,cur_ind,lambdas(1,i)/rho);
    end
    for i=1:numChunks
        Theta{i,10}=W{i,1}+Gamma{i,10}/rho;
    end
    if(l1_not_l21)
        for i=1:numChunks
            Theta{i,10} = regularizer_l1(Theta{i,10},XYZ{1,i},numX,numU,numZ,lambdas(1,10)/rho);
        end
    else
        cur_ind = {[2,numX+1],[2,numU+1],[2,numZ+1]};
        Theta(:,10) = regularizer_l21(Theta(:,10),XYZ,cur_ind,lambdas(1,10)/rho);
    end
    for i=1:numChunks
        for j=1:numCoefs
            Gamma{i,j}=Gamma{i,j}+rho*(W{i,1}-Theta{i,j});
        end
    end
    s_mat = 0;
    r_mat = 0;
    for i=1:numChunks
        for j=1:numCoefs
            s_mat = s_mat + sum(sum(sum((rho*(Theta{i,j}-Theta_old{i,j})).^2)));
            r_mat = r_mat + sum(sum(sum((rho*(W{i,1}-Theta{i,j})).^2)));
        end
    end
    s = sqrt(s_mat);
    r = sqrt(r_mat);
    if(r>10*s)
        rho = 2*rho;
    else
        if(10*r<s)
            rho = rho/2;
        end
    end
    fprintf('r:%e\ts:%e\trho:%f\n',r,s,rho);
    if(r < eps_pri && s < eps_dual)
        break;
    end
end
Ws = cell(numChunks,1);
for i=1:numChunks
    cur_W = {W{i,1},XYZ{1,i}{1,2}};
    Ws{i,1} = cur_W;
end
%imagesc(abs(B_full));
end
function [Ws]=regularizer_l21(Ws,XYZs,x_y_z_ind,lambd)
    [x_ind,y_ind,z_ind] = deal(x_y_z_ind{:});
    sub_sizes=[x_ind(1,2)-x_ind(1,1)+1,y_ind(1,2)-y_ind(1,1)+1,z_ind(1,2)-z_ind(1,1)+1];
    numChunks = size(Ws,1);
    X = zeros(sub_sizes(1,1)*sub_sizes(1,2)*sub_sizes(1,3),numChunks);
    [numX1,numU1,numZ1] = size(Ws{1,1});
    sup_mat = zeros(numChunks,numX1,numU1,numZ1);
    sup_mat_ind = zeros(numChunks,numX1,numU1,numZ1);
    for i=1:numChunks
        sup_mat(i,:,:,:) = Ws{i,1};
        [x_i,y_i,z_i] = deal(XYZs{1,i}{1,2}{:});
        sup_mat_ind(i,x_i(1,1):x_i(1,2),y_i(1,1):y_i(1,2),z_i(1,1):z_i(1,2)) = 1;
    end
    for i=x_ind(1,1):x_ind(1,2)
        for j=y_ind(1,1):y_ind(1,2)
            for k=z_ind(1,1):z_ind(1,2)
                cur_vec = zeros(1,numChunks);
                cur_ind = zeros(1,numChunks);
                cur_vec(1,:) = sup_mat(:,i,j,k);
                cur_ind(1,:) = sup_mat_ind(:,i,j,k);
                cur_X = cur_vec(1,cur_ind==1);
                param.g_d = 1:size(cur_X,2);
                param.g_t = size(cur_X,2);
                [~,tmp_res] = evalc('prox_l21(cur_X,lambd/numChunks,param)');
                cur_X1 = tmp_res;
                cur_vec(1,cur_ind==1) = cur_X1;
                sup_mat(:,i,j,k) = cur_vec;
            end
        end
    end
    for i=1:numChunks
        Ws{i,1}(:,:,:) = sup_mat(i,:,:,:);
    end
end
function theta = regularizer_l1(theta,XYZ,numX,numU,numZ,tmp_lambda_rho)
    [x_i,y_i,z_i] = deal(XYZ{1,2}{:});
    x_i = x_i - 1;x_i(1,1)=max(x_i(1,1),1);x_i(1,2)=min(x_i(1,2),numX);
    y_i = y_i - 1;y_i(1,1)=max(y_i(1,1),1);y_i(1,2)=min(y_i(1,2),numU);
    z_i = z_i - 1;z_i(1,1)=max(z_i(1,1),1);z_i(1,2)=min(z_i(1,2),numZ);
    tmp_theta = theta(2:numX+1,2:numU+1,2:numZ+1);
    for i=x_i(1,1):x_i(1,2)
        for j=y_i(1,1):y_i(1,2)
            for k=z_i(1,1):z_i(1,2)
                tmp_theta(i,j,k)=sign(tmp_theta(i,j,k))*max(abs(tmp_theta(i,j,k))-tmp_lambda_rho,0);
            end
        end
    end
    theta(2:numX+1,2:numU+1,2:numZ+1) = tmp_theta;
end
function [val] = logit_fun(y,xb)
    val = ones(size(xb))./(1+exp(-xb))-y;
end
function [Ws1]=regularizer(Ws,XYZ,x_y_z_ind,lambd)
    [x_ind,y_ind,z_ind] = deal(x_y_z_ind{:});
    [num_x,num_y,num_z] = size(Ws{1,1});
    sub_sizes=[x_ind(1,2)-x_ind(1,1)+1,y_ind(1,2)-y_ind(1,1)+1,z_ind(1,2)-z_ind(1,1)+1];
    if(x_ind(1,1)==1&&x_ind(1,2)==num_x) order=[1,2,3];re_order = [1,2,3];
    elseif(y_ind(1,1)==1&&y_ind(1,2)==num_y) order=[2,1,3];re_order = [2,1,3];
    elseif(z_ind(1,1)==1&&z_ind(1,2)==num_z) order=[3,1,2];re_order = [2,3,1];
    else error('dimension error'); end
    Xs = {};
    numChunks = size(Ws,1);
    for i=1:numChunks
        x_y_z_i = XYZ{1,i}{1,2};
        [tmp_X,ind_W] = preprocess(Ws{i,1},order,sub_sizes,x_y_z_ind,x_y_z_i);
        Xs = {Xs{:},{tmp_X,ind_W}};
    end
    Xs = cal_l21(Xs,lambd);
    Ws1 = {};
    for i=1:numChunks
        tmp_W = postprocess(Ws{i,1},Xs{1,i}{1,1},order,re_order,sub_sizes,x_y_z_ind);
        Ws1 = {Ws1{:},tmp_W};
    end
end
function W = postprocess(W,X,order,re_order,sub_sizes,x_y_z_ind)
    [x_ind,y_ind,z_ind] = deal(x_y_z_ind{:});
    X=reshape(X,sub_sizes(1,order(1,1)),sub_sizes(1,order(1,2)),sub_sizes(1,order(1,3)));
    X=permute(X,re_order);
    W(x_ind(1,1):x_ind(1,2),y_ind(1,1):y_ind(1,2),z_ind(1,1):z_ind(1,2))=X;
end
function Xs1 = cal_l21(Xs,lambd)
numChunks = size(Xs,2);
numCols = size(Xs{1,1}{1,1},2);
numRows = size(Xs{1,1}{1,1},1);
X_vecs = {};
X_mat = zeros(numRows*numChunks,numCols);
X_mat1 = zeros(numRows*numChunks,numCols);
I_mat = zeros(numRows*numChunks,numCols);
for i=1:numChunks
    X_mat((i-1)*numRows+1:i*numRows,:)=Xs{1,i}{1,1};
    I_mat((i-1)*numRows+1:i*numRows,:)=Xs{1,i}{1,2};
end
for i=1:numCols
    param.g_d = 1:sum(I_mat(:,i));
    param.g_t = [sum(I_mat(:,i))];
    [~,tmp_res] = evalc('prox_l21(X_mat(I_mat(:,i)==1,i),lambd/numCols,param)');
    X_mat1(I_mat(:,i)==1,i) = tmp_res;
end
Xs1 = cell(1,numChunks);
for i=1:numChunks
    Xs1{1,i}={X_mat1((i-1)*numRows+1:i*numRows,:),Xs{1,i}{1,2}};
end
end
function [X,X0] = preprocess(W,order,sub_sizes,x_y_z_ind,x_y_z_i)
    [x_ind,y_ind,z_ind] = deal(x_y_z_ind{:});
    [x_i,y_i,z_i] = deal(x_y_z_i{:});
    X = W(x_ind(1,1):x_ind(1,2),y_ind(1,1):y_ind(1,2),z_ind(1,1):z_ind(1,2));
    X0 = zeros(size(X));
    X0(max(x_i(1,1)-x_ind(1,1)+1,1):min(x_i(1,2)-x_ind(1,1)+1,x_ind(1,2)-x_ind(1,1)+1),...
        max(y_i(1,1)-y_ind(1,1)+1,1):min(y_i(1,2)-y_ind(1,1)+1,y_ind(1,2)-y_ind(1,1)+1),...
        max(z_i(1,1)-z_ind(1,1)+1,1):min(z_i(1,2)-z_ind(1,1)+1,z_ind(1,2)-z_ind(1,1)+1))=1;
    X=permute(X,order);
    X0=permute(X0,order);
    X=reshape(X,sub_sizes(1,order(1,1)),sub_sizes(1,order(1,2))*sub_sizes(1,order(1,3)));
    X0=reshape(X0,sub_sizes(1,order(1,1)),sub_sizes(1,order(1,2))*sub_sizes(1,order(1,3)));
end

function tmp_B = update_B_log(Y,current_b,w,s,v,u,M,coef,rh,numObs)
    ITER_MAX2 = 100;
    B_TOL = 1e-4;
    d = sum(s,2);
    w_current_b = w*current_b;
    resp_vec1 = (1/(2*sqrt(numObs)))*(w_current_b - 4*logit_fun(Y,w_current_b));
    resp_vec2 = M'/sqrt(coef*rh);
    sec_part = v'*resp_vec2;
    sec_part = d.^2./(coef*4*numObs*rh+d.^2).*sec_part;
    sec_part = v*sec_part;
    sec_part = (resp_vec2 - sec_part)/sqrt(coef*rh);
    for i=1:ITER_MAX2
        first_part = d/(2*sqrt(numObs)).*(u'*resp_vec1);
        first_part = first_part - (d.^2./(12*numObs*rh+d.^2)).*first_part;
        first_part = v*first_part/(coef*rh);
        b_new = first_part + sec_part;
        if(all(abs(current_b-b_new)<B_TOL))
            break
        else
            current_b = b_new;
            w_new_b = w*b_new;
            resp_vec1 = (1/(2*sqrt(numObs)))*(w_new_b - 4*logit_fun(Y,w_new_b));
        end
    end
    tmp_B = b_new;
end
function [b0,ls_dict] = update_B(Theta,Gamma,rho,linear_not_log,XYZ,ls_dict,numCoefs,numX,numU,numZ)
    b0 = zeros(numX+1,numU+1,numZ+1);
    [Y,XZ_T_XZ,XZ_T_YY,w,u,s,v] = deal(XYZ{1,1}{:});
    [x_ind,u_ind,z_ind] = deal(XYZ{1,2}{:});
    numX1 = x_ind(1,2)-x_ind(1,1)+1;
    numU1 = u_ind(1,2)-u_ind(1,1)+1;
    numZ1 = z_ind(1,2)-u_ind(1,1)+1;
    b = zeros(numX1,(numU1),(numZ1));
    numVariables1 = numX1*numU1*numZ1;
    numObservations = size(Y,1);
    sumTheta = zeros(x_ind(1,2)-x_ind(1,1)+1,u_ind(1,2)-u_ind(1,1)+1,z_ind(1,2)-u_ind(1,1)+1);
    sumGamma = zeros(x_ind(1,2)-x_ind(1,1)+1,u_ind(1,2)-u_ind(1,1)+1,z_ind(1,2)-u_ind(1,1)+1);
    for i = 1:numCoefs
        sumTheta = sumTheta + Theta{1,i}(x_ind(1,1):x_ind(1,2),u_ind(1,1):u_ind(1,2),z_ind(1,1):z_ind(1,2));
        sumGamma = sumGamma + Gamma{1,i}(x_ind(1,1):x_ind(1,2),u_ind(1,1):u_ind(1,2),z_ind(1,1):z_ind(1,2));
    end
    M = (rho*sumTheta-sumGamma);
    M = reshape(M,1,[]);
    if(linear_not_log == true)
        Right = XZ_T_YY+numObservations*M';
        if isKey(ls_dict,num2str(rho))
            Left_inv = ls_dict(num2str(rho));
        else
            Left_inv = inv(XZ_T_XZ +numCoefs*rho*numObservations*eye(numVariables1));
            ls_dict(num2str(rho))=Left_inv;
        end
        tmp_B = Left_inv*Right;
    else
        current_b = reshape(b,1,numVariables1)';
        tmp_B = update_B_log(Y,current_b,w,s,v,u,M,numCoefs,rho,numObservations);
    end
    b = reshape(tmp_B,numX1,numU1,numZ1);
    b0(x_ind(1,1):x_ind(1,2),u_ind(1,1):u_ind(1,2),z_ind(1,1):z_ind(1,2))=b;
end