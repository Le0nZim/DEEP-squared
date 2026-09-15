function depth_generate(config_file)
% Noninteractive entry point called by python depth_study.py prepare/run.
c=jsondecode(fileread(config_file));
fwd=fullfile(c.repo_root,'fwd_model');
addpath(fwd,fullfile(fwd,'_submodules','MC_LightScattering'),...
    fullfile(fwd,'_submodules','MC_LightScattering','_supToolboxes','optical_PSF'));
assert(license('test','Image_Toolbox'),'Image Processing Toolbox is required');
assert(license('test','Statistics_Toolbox'),'Statistics and Machine Learning Toolbox is required');
assert(license('test','Distrib_Computing_Toolbox'),'Parallel Computing Toolbox is required');
gpuDevice(c.optics.gpu_index);
depth_native_checks(c);
assert(c.optics.axial_planes==round(c.optics.axial_planes/2)*2,'axial_planes must be even');
assert(abs(sum(c.data.split_fractions)-1)<1e-8,'Split fractions must sum to one');
if ~isfolder(c.run_dir), mkdir(c.run_dir); end
base=f_pram_init(); base.Nt=32; base.asset_dir=c.data_dir; base.dz=c.data.dz_um;
base.pattern_typ='dmd_exp_tfm_mouse_20201224_100um';
[E,~,~,base]=f_get_extPettern(base);
[E,base]=crop_inputs(E,base,c);
optical=depth_optics(c,base);
manifest=depth_objects(c,base,optical);
source_gain=1;
if strcmp(c.signal_mode,'fixed_source')
    calibration=fullfile(c.run_dir,'source_gain.mat');
    if isfile(calibration)
        s=load(calibration,'source_gain'); source_gain=s.source_gain;
    else
        % One fixed gain: median peak of first 16 TRAINING objects under old
        % PSFs at 2 SLS. Applied unchanged at all depths and to both variants.
        [P,~]=depth_psfs(c,base,optical,'legacy',2,'train');
        P=depth_prepare_forward(P,E,base);
        n=min(16,c.data.counts.train); peaks=zeros(n,1);
        for i=1:n
            X=h5read(fullfile(c.run_dir,'objects','train.h5'),'/object',[1 1 1 i],...
                [base.Nx base.Ny manifest.object_nz 1]);
            [y,~]=depth_forward(permute(X,[2 1 3]),E,P,base); peaks(i)=max(y(:));
        end
        source_gain=base.maxcount/median(peaks);
        tmp=[calibration '.partial.mat']; save(tmp,'source_gain','peaks'); movefile(tmp,calibration,'f');
        clear P
    end
end
coverage=struct('depth_sls',{},'experimental_available',{},'calibration_reference_sls',{});
for depth=c.depths_sls(:)'
    pram=f_pram_init(); pram.Nt=32; pram.asset_dir=c.data_dir; pram.dz=c.data.dz_um;
    um=depth*1e4/c.optics.mus_cm_inv;
    available=ismember(um,[100 200 300 350 400]);
    if available
        reference_depth=depth;
    else
        reference_depth=c.extension.unmeasured_peak_reference_sls;
    end
    pram.pattern_typ=sprintf('dmd_exp_tfm_mouse_20201224_%gum',reference_depth*50);
    [Ed,Yexp,~,pram]=f_get_extPettern(pram);
    [Ed,pram,yr,xr]=crop_inputs(Ed,pram,c);
    assert(isequal(Ed,E),'Pattern identity changed across depths');
    pram.z0_um=-um; % Never overwrite requested depth with a calibration depth.
    pram.emhist_dir=fullfile(c.run_dir,'camera_lut');
    if available
        write_experimental(c,depth,Yexp,pram,yr,xr);
    end
    info=struct('depth_sls',depth,'requested_depth_um',um,'calibration_reference_sls',reference_depth,...
        'signal_mode',c.signal_mode,'peak_target_electrons',pram.maxcount,...
        'source_gain',source_gain,'camera',pram,...
        'pattern_ids_matlab',21:52,'note',c.extension.note);
    file=fullfile(c.run_dir,'calibration',sprintf('%gsls.json',depth));
    if ~isfolder(fileparts(file)), mkdir(fileparts(file)); end
    fid=fopen(file,'w'); fprintf(fid,'%s\n',jsonencode(info)); fclose(fid);
    variants={'legacy','corrected'};
    for arm=1:2
        variant=variants{arm};
        [P,~]=depth_psfs(c,pram,optical,variant,depth,'train');
        P=depth_prepare_forward(P,E,pram);
        depth_write_data(c,pram,E,P,manifest,variant,depth,'train',source_gain);
        depth_write_data(c,pram,E,P,manifest,variant,depth,'val',source_gain);
        if c.mc.independent_test_psfs
            clear P
            [P,~]=depth_psfs(c,pram,optical,variant,depth,'test');
            P=depth_prepare_forward(P,E,pram);
        end
        depth_write_data(c,pram,E,P,manifest,variant,depth,'test',source_gain);
        clear P
    end
    coverage(end+1)=struct('depth_sls',depth,'experimental_available',available,'calibration_reference_sls',reference_depth); %#ok<AGROW>
end
fid=fopen(fullfile(c.run_dir,'generation_coverage.json'),'w'); fprintf(fid,'%s\n',jsonencode(coverage)); fclose(fid);
fprintf('All requested native MATLAB datasets generated: %s\n',c.run_dir);
end

function [E,p,yr,xr]=crop_inputs(E,p,c)
n=c.data.crop_size;
if n==0, n=min(size(E,1),size(E,2)); end
assert(n>=32 && n<=min(size(E,1),size(E,2)),'crop_size is invalid');
yr=floor((size(E,1)-n)/2)+(1:n); xr=floor((size(E,2)-n)/2)+(1:n);
E=E(yr,xr,:); p.Nx=n; p.Ny=n;
end

function write_experimental(c,depth,Y,p,yr,xr)
folder=fullfile(c.run_dir,'experimental'); if ~isfolder(folder), mkdir(folder); end
file=fullfile(folder,sprintf('%gsls.h5',depth)); if isfile(file), return; end
names=fieldnames(Y); n=numel(names);
partial=[file '.partial']; if isfile(partial), delete(partial); end
h5create(partial,'/input',[p.Nx p.Ny 32 n],'Datatype','single','ChunkSize',[p.Nx p.Ny 1 1],'Deflate',4);
h5create(partial,'/sample_id',[1 n],'Datatype','int64');
for i=1:n
    value=Y.(names{i}); value=value(yr,xr,:);
    assert(size(value,3)==32 && all(isfinite(value(:))),'Invalid experimental stack');
    h5write(partial,'/input',permute(value,[2 1 3]),[1 1 1 i],[p.Nx p.Ny 32 1]);
    h5write(partial,'/sample_id',int64(1000000+round(depth*100)+i),[1 i],[1 1]);
end
h5writeatt(partial,'/','experiment_id',c.experiment_id);
h5writeatt(partial,'/','complete',1); h5writeatt(partial,'/','depth_sls',depth);
h5writeatt(partial,'/','field_names',jsonencode(names));
h5writeatt(partial,'/','units','Input-equivalent intensity as returned by upstream f_get_extPettern; no new ADU conversion');
h5writeatt(partial,'/','pattern_ids_matlab',int32(21:52));
h5writeatt(partial,'/','gt_note','No matched ground truth; X_refs and widefield images are not used as ground truth');
movefile(partial,file,'f');
end
