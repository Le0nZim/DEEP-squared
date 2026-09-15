function Y = depth_noise(Y0,pram,seed,camera_mode,read_seed)
% Paired camera ablation. The legacy Gaussian is shared by an explicitly
% recorded generation batch; the corrected Gaussian is element independent.
% Separate read/shot streams make resumes independent of table cache history.
persistent legacy_table
p=pram; p.cam_sigma_rd=0;
rng(seed,'twister');
if strcmp(camera_mode,'legacy')
    if isempty(legacy_table)
        s=load(fullfile(pram.repo_root,'fwd_model','_emhist','emhist_29-Apr-2021_02_09_25.mat'),'emhist');
        legacy_table=s.emhist;
    end
    % Byte-exact historical shot/EM path, including the 100-count cap and
    % fixed supplied LUT. Apply the old batch-wide Gaussian below so chunks
    % do not change its correlation or require a full batch in RAM.
    Y=legacy_camera.f_simulateIm_emCCD(double(Y0),legacy_table,p);
elseif strcmp(camera_mode,'corrected')
    counts=poissrnd(double(Y0)+p.cam_dXdt_dark*p.cam_t_exp);
    values=depth_emhist(p,max(counts(:)));
    table=struct('values',values,'cam_N_gainStages',p.cam_N_gainStages,...
        'cam_Brnuli_alpha',p.cam_Brnuli_alpha);
    rng(seed,'twister'); % Replay counts after determining required table size.
    Y=f_simulateIm_emCCD(double(Y0),table,p);
else
    error('DEEP2:CameraMode','Unknown camera mode: %s',camera_mode);
end
rng(read_seed,'twister');
if strcmp(camera_mode,'legacy')
    noise=randn();
else
    noise=randn(size(Y));
end
Y=single(Y+pram.cam_sigma_rd/pram.cam_EMgain*noise);
assert(all(isfinite(Y(:))),'Nonfinite noisy measurement');
end
