function [XUZ,XUZ_std] = data_precompute(X,U,Z,varargin)
numX = size(X,2);
numU = size(U,2);
numZ = size(Z,2);

numObservations = size(X,1);
numFeatures = (numX+1)*(numU+1)*(numZ+1);
reshp_index = reshape(1:numFeatures,numX+1,numU+1,numZ+1);
i_j_vs = zeros(3,0);
for cur_index=1:numObservations
    cur_i_j_vs = zeros(3,0);
        vec_X = X(cur_index,:);
        vec_U = U(cur_index,:);
        vec_Z = Z(cur_index,:);
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,1,1,1,1)];
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,2,1,1,vec_X)];
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,1,2,1,vec_U)];
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,1,1,2,vec_Z)];
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,2,2,1,vec_X'*vec_U)];
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,2,1,2,vec_X'*vec_Z)];
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,1,2,2,vec_U'*vec_Z)];
        cur_i_j_vs = [cur_i_j_vs,append_tensor(reshp_index,cur_index,2,2,2,kron(vec_X'*vec_U,vec_Z))];
    i_j_vs = [i_j_vs,cur_i_j_vs];
end
XUZ = sparse(i_j_vs(1,:),i_j_vs(2,:),i_j_vs(3,:),numObservations,numFeatures);
if nargin>3
    pre_stds = varargin{1};
    [XUZ,XUZ_std] = get_unit_var(XUZ,1,pre_stds);
else
    [XUZ,XUZ_std] = get_unit_var(XUZ,1);
end
end

function res_index = append_tensor(i_tensor,cur_is,x_from,y_from,z_from,cur_vs)
[numXX,numUU,numZZ] = size(i_tensor);
numXX = numXX - 1;
numUU = numUU - 1;
numZZ = numZZ - 1;
if x_from > 1 && y_from > 1 && z_from > 1
    ind_mat = zeros(size(cur_vs));
    for ii=z_from:numZZ+1
        ind_mat(:,(ii-2)*numUU+1:(ii-1)*numUU)=i_tensor(x_from:end,y_from:end,ii);
    end
else
    ind_mat = reshape(i_tensor(identify_from(x_from,numXX+1),identify_from(y_from,numUU+1),identify_from(z_from,numZZ+1)),size(cur_vs));
end
[iis,jjs,vvs] = find(cur_vs);

num_ii = max(size(iis,1),size(iis,2));
if min(size(iis,1),size(iis,2))~=0
    iis = reshape(iis,1,num_ii);
    jjs = reshape(jjs,1,num_ii);
    vvs = reshape(vvs,1,num_ii);
    ijs = zeros(1,num_ii);
    for ii = 1:num_ii
        ijs(1,ii)=ind_mat(iis(1,ii),jjs(1,ii));
    end
    res_index = [ones(size(ijs))*cur_is;ijs;vvs];
else
    res_index = zeros(3,0);
end
end

function x_index = identify_from(x_from,x_size)
if(x_from==1)
    x_index = 1;
else
    x_index = x_from:x_size;
end
end
