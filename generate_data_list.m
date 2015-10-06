data_list = cell(4,1);
num = zeros(4,1);
for i = iter_num : -1 : 1
data = load(['data/' num2str(i) '.mat']);
data_list{data.action_id_t} = [data_list{data.action_id_t} i];
num(data.action_id_t) = num(data.action_id_t)+1;
if mod(i,100)==0 fprintf('i = %d\n',i); end
if min(num) > buffer_sz
    break;
end

end