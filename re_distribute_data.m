function [XYs] = re_distribute_data(X,Y,sizes,splits)
% splits = {struct('time',[1,20],'value',struct('x',[1,1],'u',[1,Inf],'z',[1,Inf])),...
% struct('time',[21,40],'value',struct('x',[1,Inf],'u',[1,Inf],'z',[1,Inf])),...
% struct('time',[41,60],'value',struct('x',[1,Inf],'u',[1,Inf],'z',[1,1]))};

% splits = {struct('time',[1,20],'value',struct('x',[1,Inf],'u',[1,1],'z',[1,Inf])),...
% struct('time',[21,40],'value',struct('x',[1,Inf],'u',[1,Inf],'z',[1,Inf])),...
% struct('time',[41,60],'value',struct('x',[1,Inf],'u',[1,Inf],'z',[1,1]))};

% sizes = struct('numDates',275,'numX',234,'numU',3,'numZ',4);
numObservations = size(X,1);
if(numObservations ~= size(Y,1))
    error('dimension inconsistent');
end
% splits = {[1,100],[101,200],[201,275]};
numDates = sizes.numDates;
numX = sizes.numX;
numU = sizes.numU;
numZ = sizes.numZ;
numChunks = size(splits,2);
if(mod(numObservations,numDates)~=0)
    error('numDates error');
end
numCities = numObservations/numDates;
XYs = {};
for i=1:numChunks
    i
    cur_mat = zeros(numX+1,numU+1,numZ+1);
    cur_ind = splits{1,i};
    ind_x = cur_ind.value.x;
    ind_u = cur_ind.value.u;
    ind_z = cur_ind.value.z;
    cur_mat(ind_x(1,1):min(ind_x(1,2),numX+1),ind_u(1,1):min(ind_u(1,2),numU+1),ind_z(1,1):min(ind_z(1,2),numZ+1)) = 1;
    cur_vec = reshape(cur_mat,1,[]);
    cur_X = X((cur_ind.time(1,1)-1)*numCities+1:cur_ind.time(1,2)*numCities,cur_vec==1);
    cur_Y = Y((cur_ind.time(1,1)-1)*numCities+1:cur_ind.time(1,2)*numCities,:);
    [input] = pre_compute_input_v3(cur_X,cur_Y,0.5);
    re_ind = {[ind_x(1,1),min(ind_x(1,2),numX+1)],[ind_u(1,1),min(ind_u(1,2),numU+1)],[ind_z(1,1),min(ind_z(1,2),numZ+1)]};
    XYs = {XYs{:},{input,re_ind}};
end
end
