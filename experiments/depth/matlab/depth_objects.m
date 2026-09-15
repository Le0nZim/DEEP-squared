function manifest = depth_objects(c,pram,optical)
% Split contiguous source-z blocks BEFORE selecting patches, with a guard band.
% All depths and both PSF variants use this one persisted object manifest.
folder=fullfile(c.run_dir,'objects'); if ~isfolder(folder), mkdir(folder); end
manifest_file=fullfile(folder,'manifest.mat');
if isfile(manifest_file)
    s=load(manifest_file,'manifest'); manifest=s.manifest; return
end
s=load(fullfile(c.data_dir,'BV_03102021.mat'),'Data');
assert(isfield(s.Data,'cell') && isfield(s.Data,'pram'),'Expected Data.cell and Data.pram in BV_03102021.mat');
raw=single(s.Data.cell);
source_dx=double(s.Data.pram.dx);
% Paper, Mouse cortical vasculature dataset: axial step 1.5 um.
source_dz=c.data.source_dz_um;
shape=[round(size(raw,1)*source_dx/pram.dx),round(size(raw,2)*source_dx/pram.dx),round(size(raw,3)*source_dz/pram.dz)];
V=imresize3(raw,shape,'linear','Antialiasing',true); clear raw s
V=max(V,0);
training_end=round(c.data.split_fractions(1)*size(V,3));
amplitude=max(V(:,:,1:training_end),[],'all');
assert(amplitude>0,'Training source volume is empty');
V=V/amplitude;
scaled=imresize3(optical.exPSF,'Scale',[optical.pram.dx/pram.dx,optical.pram.dx/pram.dx,optical.pram.dx/pram.dz],'Method','linear','Antialiasing',true);
object_nz=size(scaled,3);
assert(size(V,1)>=pram.Ny && size(V,2)>=pram.Nx,'Source volume is smaller than requested patch');
split_names={'train','val','test'};
cuts=[0 round(cumsum(c.data.split_fractions(:)')*size(V,3))];
cuts(end)=size(V,3);
manifest=struct('objects',[],'source_dx_um',source_dx,'source_dz_um',source_dz,...
    'output_dx_um',pram.dx,'output_dz_um',pram.dz,'object_nz',object_nz,...
    'split_description','Disjoint contiguous axial blocks with two-plane boundary guards; within-volume generalization, not held-out animals');
sample_id=0;
for sp=1:3
    name=split_names{sp}; count=c.data.counts.(name);
    zmin=cuts(sp)+3; zmax=cuts(sp+1)-2;
    assert(zmax-zmin+1>=object_nz,'Split is too small for the optical axial support');
    rng(c.data.seed+sp,'twister');
    h5file=fullfile(folder,[name '.h5']);
    partial=[h5file '.partial']; if isfile(partial), delete(partial); end
    h5create(partial,'/object',[pram.Nx pram.Ny object_nz count],...
        'Datatype','single','ChunkSize',[pram.Nx pram.Ny object_nz 1],'Deflate',4);
    % Record duplicate source coordinates; no repeats within a split.
    used=containers.Map('KeyType','char','ValueType','logical');
    for i=1:count
        found=false;
        for attempt=1:10000
            yy=randi(size(V,1)-pram.Ny+1); xx=randi(size(V,2)-pram.Nx+1);
            zz=randi([zmin,zmax-object_nz+1]);
            key=sprintf('%d_%d_%d',yy,xx,zz);
            X=V(yy:yy+pram.Ny-1,xx:xx+pram.Nx-1,zz:zz+object_nz-1);
            if max(X(:))>0 && ~isKey(used,key), found=true; used(key)=true; break; end
        end
        assert(found,'Cannot find enough nonempty distinct source patches');
        sample_id=sample_id+1;
        % MATLAB HDF5 dimensions are reversed in Python. Explicit x/y transpose
        % makes the eventual Python layout [N,Z,Y,X], without relying on square images.
        h5write(partial,'/object',permute(X,[2 1 3]),[1 1 1 i],[pram.Nx pram.Ny object_nz 1]);
        entry=struct('sample_id',sample_id,'split',name,'index',i,'x',xx,'y',yy,...
            'z_first',zz,'z_last',zz+object_nz-1,'rotation_degrees',0);
        manifest.objects=[manifest.objects; entry]; %#ok<AGROW>
    end
    h5writeatt(partial,'/','experiment_id',c.experiment_id);
    h5writeatt(partial,'/','complete',1); movefile(partial,h5file,'f');
end
temp=[manifest_file '.partial.mat']; save(temp,'manifest'); movefile(temp,manifest_file,'f');
fid=fopen(fullfile(folder,'manifest.json'),'w'); fprintf(fid,'%s\n',jsonencode(manifest)); fclose(fid);
end
