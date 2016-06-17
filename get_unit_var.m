function [ax,stds] = get_unit_var(x,except_c,varargin)
[m,n] = size(x);
ax = sparse(m,n);
v_i = [];
v_j = [];
v_s = [];
pre_stds = false;
if nargin>2
    pre_stds = varargin{1};
end
count = 0;
stds = [];
for i=1:size(x,2)
    cur_x = x(:,i);
    if(i==except_c)
        [cur_v_i,cur_v_j,cur_v_s] = find(cur_x);
        v_i = [v_i;cur_v_i];
        v_j = [v_j;repmat(i,size(cur_v_i))];
        v_s = [v_s;cur_v_s];
        stds = [stds,1];
        continue
    end
    count = count + 1;
    if(sum(cur_x)==0)
        ax(:,i) = cur_x;
        stds = [stds,1];
    else
        if pre_stds == false
            stdx = std(cur_x);
        else
            stdx = pre_stds(1,count);
        end
        stds = [stds,stdx];
        cur_mat = (cur_x)./stdx(ones(m,1),:);
        [cur_v_i,cur_v_j,cur_v_s] = find(cur_mat);
        v_i = [v_i;cur_v_i];
        v_j = [v_j;repmat(i,size(cur_v_i))];
        v_s = [v_s;cur_v_s];
        % ax(:,i) = ;
    end
end
ax = sparse(v_i,v_j,v_s,m,n);
end