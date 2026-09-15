function depth_write_data(c,pram,E,PSFs,manifest,variant,depth,split,source_gain)
folder=fullfile(c.run_dir,'datasets',sprintf('%gsls',depth),variant);
if ~isfolder(folder), mkdir(folder); end
file=fullfile(folder,[split '.h5']);
if isfile(file), return; end
partial=[file '.partial'];
entries=manifest.objects(strcmp({manifest.objects.split},split));
n=numel(entries); done=0;
if isfile(partial)
    done=h5readatt(partial,'/','written_samples');
else
    h5create(partial,'/input',[pram.Nx pram.Ny 32 n],'Datatype','single',...
        'ChunkSize',[pram.Nx pram.Ny 1 1],'Deflate',4);
    h5create(partial,'/gt',[pram.Nx pram.Ny 1 n],'Datatype','single',...
        'ChunkSize',[pram.Nx pram.Ny 1 1],'Deflate',4);
    h5create(partial,'/sample_id',[1 n],'Datatype','int64');
    h5create(partial,'/noiseless_peak',[1 n],'Datatype','single');
    h5create(partial,'/signal_multiplier',[1 n],'Datatype','double');
    h5writeatt(partial,'/','written_samples',0);
    h5writeatt(partial,'/','experiment_id',c.experiment_id);
    h5writeatt(partial,'/','variant',variant); h5writeatt(partial,'/','depth_sls',depth);
    h5writeatt(partial,'/','signal_mode',c.signal_mode); h5writeatt(partial,'/','complete',0);
end
object_file=fullfile(c.run_dir,'objects',[split '.h5']);
for i=done+1:n
    X=h5read(object_file,'/object',[1 1 1 i],[pram.Nx pram.Ny manifest.object_nz 1]);
    X=permute(X,[2 1 3]);
    [Y0,gt]=depth_forward(X,E,PSFs,pram);
    if strcmp(c.signal_mode,'paper_peak')
        multiplier=pram.maxcount/double(max(Y0(:)));
    else
        multiplier=source_gain;
    end
    Y0=double(Y0)*multiplier;
    seed=mod(c.data.seed+entries(i).sample_id+round(depth*100000),2^32-1);
    Y=depth_noise(Y0,pram,seed);
    h5write(partial,'/input',permute(Y,[2 1 3]),[1 1 1 i],[pram.Nx pram.Ny 32 1]);
    h5write(partial,'/gt',gt',[1 1 1 i],[pram.Nx pram.Ny 1 1]);
    h5write(partial,'/sample_id',int64(entries(i).sample_id),[1 i],[1 1]);
    h5write(partial,'/noiseless_peak',single(max(Y0(:))),[1 i],[1 1]);
    h5write(partial,'/signal_multiplier',multiplier,[1 i],[1 1]);
    h5writeatt(partial,'/','written_samples',i);
    fprintf('DATA %g SLS %s %s %d/%d\n',depth,variant,split,i,n);
end
h5writeatt(partial,'/','complete',1); movefile(partial,file,'f');
end
