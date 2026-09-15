function table=depth_emhist(p,max_count)
% Finite empirical conditional distribution, like upstream emhist; no count cap.
persistent cache_key cached_table
key=sprintf('stages%d_alpha%.8g',p.cam_N_gainStages,p.cam_Brnuli_alpha);
if isempty(cache_key) || ~strcmp(cache_key,[p.emhist_dir key])
    if ~isfolder(p.emhist_dir), mkdir(p.emhist_dir); end
    file=fullfile(p.emhist_dir,[key '.mat']);
    if isfile(file)
        s=load(file,'table'); cached_table=s.table;
    else
        cached_table=zeros(0,10000);
    end
    cache_key=[p.emhist_dir key];
end
table=cached_table;
if max_count<=size(table,1), return; end
first=size(table,1)+1;
for n=first:max_count
    rng(mod(271828+p.cam_N_gainStages*10000+n,2^32-1),'twister');
    row=n*ones(1,10000);
    for stage=1:p.cam_N_gainStages
        row=row+binornd(row,p.cam_Brnuli_alpha);
    end
    table(n,:)=row; %#ok<AGROW>
end
cached_table=table;
file=fullfile(p.emhist_dir,[key '.mat']); temporary=[file '.partial.mat'];
save(temporary,'table'); movefile(temporary,file,'f');
fprintf('Camera conditional LUT: %s, input counts 0:%d (no clipping)\n',key,size(table,1));
end
