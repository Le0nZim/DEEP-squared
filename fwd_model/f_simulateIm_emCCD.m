function [Xhat,XhatADU] = f_simulateIm_emCCD(X0,emhist,pram)
% Independent shot/EM/read noise at every pixel, pattern and sample.
% emhist is a metadata-bearing struct from f_genEmhist, or [] for direct EM.
% Bare numeric tables cannot establish compatibility with the configured gain.
was_gpu=isa(X0,'gpuArray');
if was_gpu, X0=gather(X0); end
X0=double(X0);
assert(~isempty(X0) && isreal(X0) && all(isfinite(X0(:))) && all(X0(:)>=0),...
    'DEEP2:InvalidPhotonRate','Expected finite nonnegative input photon rates');
validateattributes(pram.cam_dXdt_dark,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(pram.cam_t_exp,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(pram.cam_sigma_rd,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(pram.cam_EMgain,{'numeric'},{'scalar','finite','positive'});
validateattributes(pram.cam_N_gainStages,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(pram.cam_Brnuli_alpha,{'numeric'},{'scalar','>=',0,'<=',1});
if ~isempty(emhist)
    assert(isstruct(emhist) && all(isfield(emhist,{'values','cam_N_gainStages','cam_Brnuli_alpha'})),...
        'DEEP2:UnverifiedEMTable','Regenerate the numeric legacy table with f_genEmhist, or pass [] for direct EM');
    assert(emhist.cam_N_gainStages==pram.cam_N_gainStages && emhist.cam_Brnuli_alpha==pram.cam_Brnuli_alpha,...
        'DEEP2:EMGainMismatch','EM lookup table does not match the configured multiplication process');
    table=emhist.values;
    assert(ismatrix(table) && size(table,2)>0 && all(isfinite(table(:))) && all(table(:)>=0),...
        'DEEP2:InvalidEMTable','Invalid conditional electron-count table');
else
    table=zeros(0,1);
end
% CPU random sampling also avoids the old catch-all GPU fallback. A caller's
% rng seed has the same meaning whether input/output storage is CPU or GPU.
counts=poissrnd(X0+pram.cam_dXdt_dark*pram.cam_t_exp);
electrons=zeros(size(counts));
within=find(counts>0 & counts<=size(table,1));
if ~isempty(within)
    draws=randi(size(table,2),size(within));
    electrons(within)=table(sub2ind(size(table),counts(within),draws));
end
% No saturation at the lookup-table boundary: simulate the same Bernoulli
% register directly for counts outside the table (or all counts if table=[]).
outside=find(counts>size(table,1));
direct=counts(outside);
for stage=1:pram.cam_N_gainStages
    direct=direct+binornd(direct,pram.cam_Brnuli_alpha);
end
electrons(outside)=direct;
% Old: normrnd(0,sigma) produced ONE scalar, broadcast over the whole batch.
% New: randn(size(electrons)) gives one independent draw per array element.
read_noise=pram.cam_sigma_rd*randn(size(electrons));
Xhat=(electrons+read_noise)/pram.cam_EMgain;
XhatADU=Xhat*pram.cam_EMgain*pram.cam_ADCfactor+pram.cam_bias;
if was_gpu, Xhat=gpuArray(Xhat); XhatADU=gpuArray(XhatADU); end
end
