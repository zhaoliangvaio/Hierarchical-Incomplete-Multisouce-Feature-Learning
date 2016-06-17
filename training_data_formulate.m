function [inputs_tr,sizes,XUZ_std] = training_data_formulate(X,U,Z,Y,chunks,varargin)
if(nargin>5)
    pos_vs_neg_weight = varargin{1};
else
    pos_vs_neg_weight = 1;
end
sizes.numX = size(X,2);
sizes.numU = size(U,2);
sizes.numZ = size(Z,2);
[XUZ,XUZ_std] = data_precompute(X,U,Z);
[inputs_tr] = process_chunks(XUZ,Y,sizes,chunks,true,pos_vs_neg_weight);
