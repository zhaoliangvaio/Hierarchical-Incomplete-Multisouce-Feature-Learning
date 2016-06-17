function [XYs] = process_chunks(X,Y,sizes,chunks,train_not_test,pos_vs_neg_weight)
numObservations = size(X,1);
if(numObservations ~= size(Y,1))
    error('dimension inconsistent');
end
numX = sizes.numX;
numU = sizes.numU;
numZ = sizes.numZ;
numChunks = size(chunks,2);
XYs = {};
for i=1:numChunks
    cur_mat = zeros(numX+1,numU+1,numZ+1);
    cur_ind = chunks{1,i};
    ind_x = cur_ind.value.x;
    ind_u = cur_ind.value.u;
    ind_z = cur_ind.value.z;
    cur_mat(ind_x(1,1):min(ind_x(1,2),numX+1),ind_u(1,1):min(ind_u(1,2),numU+1),ind_z(1,1):min(ind_z(1,2),numZ+1)) = 1;
    cur_vec = reshape(cur_mat,1,[]);
    
    cur_Y = Y(cur_ind.time(1,1):cur_ind.time(1,2),:);
    if train_not_test
        cur_X = X(cur_ind.time(1,1):cur_ind.time(1,2),cur_vec==1);
    	[input] = compute_svd(cur_X,cur_Y,pos_vs_neg_weight);
    else
        cur_X = X(cur_ind.time(1,1):cur_ind.time(1,2),:);
        cur_X(:,cur_vec==0)=0;
        input = {cur_X,cur_Y};
    end
    re_ind = {[ind_x(1,1),min(ind_x(1,2),numX+1)],[ind_u(1,1),min(ind_u(1,2),numU+1)],[ind_z(1,1),min(ind_z(1,2),numZ+1)]};
    XYs = {XYs{:},{input,re_ind}};
end
end


function [input] = compute_svd(X,Y,pos_vs_neg_weight)
if(pos_vs_neg_weight~=1)
    [X,Y] = resampling(X,Y,pos_vs_neg_weight);
end
w = X;
[u,s,v] = svd(full(w),'econ');
X_T_Y = X'*Y;
disp('X_T_Y done');
X_T_X = X'*X;
disp('multiplication done');
input = {Y,X_T_X,X_T_Y,w,u,s,v};
end

function [X_new,Y_new] = resampling(X,Y,pos_vs_neg_weight)
pos_vs_neg_ratio = sum(Y==1)/sum(Y~=1);
pos_ratio = pos_vs_neg_ratio*pos_vs_neg_weight;
numObservations = size(Y,1);
Y_pos = Y(Y>0.5,:);
Y_neg = Y(Y<=0.5,:);
X_pos = X(Y>0.5,:);
X_neg = X(Y<=0.5,:);
sprev = rng(0,'v5uniform');
pos_ind = datasample(1:size(Y_pos,1),floor(numObservations*pos_ratio));
Y_pos = Y_pos(pos_ind,:);
X_pos = X_pos(pos_ind,:);
sprev = rng(0,'v5uniform');
neg_ind = datasample(1:size(Y_neg,1),numObservations-floor(numObservations*pos_ratio));
Y_neg = Y_neg(neg_ind,:);
X_neg = X_neg(neg_ind,:);
Y_new = [Y_pos;Y_neg];
X_new = [X_pos;X_neg];
disp('resampling done');
end

