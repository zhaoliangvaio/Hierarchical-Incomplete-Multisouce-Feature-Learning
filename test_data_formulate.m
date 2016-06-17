function [inputs_te] = test_data_formulate(X,U,Z,Y,chunks,XUZ_std)
sizes.numX = size(X,2);
sizes.numU = size(U,2);
sizes.numZ = size(Z,2);
[XUZ] = data_precompute(X,U,Z,XUZ_std);
[inputs_te] = process_chunks(XUZ,Y,sizes,chunks,false);
end