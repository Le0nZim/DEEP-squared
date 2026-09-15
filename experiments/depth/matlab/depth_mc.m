function [kernel, diagnostics] = depth_mc(c, pram, variant, z0, bank, plane, tag, hop_factor)
% Call original/corrected MATLAB transport functions, not the notebook port.
% Sum independent batches instead of retaining all photon trajectories in RAM.
if nargin<9, hop_factor=1; end
folder=fullfile(c.run_dir,'psfs',tag,variant,bank);
if ~isfolder(folder), mkdir(folder); end
file=fullfile(folder,sprintf('plane%03d_hops%d.mat',plane,hop_factor));
if isfile(file)
    s=load(file,'kernel','diagnostics'); kernel=s.kernel; diagnostics=s.diagnostics; return
end
p=f_praminit();
p.Nx=2*c.optics.sampling_factor*pram.Nx+3;
p.dx=pram.dx/c.optics.sampling_factor;
p.z0_um=z0; p.mus=c.optics.mus_cm_inv; p.sl=1e4/p.mus;
p.NA=c.optics.na;
p.Nphotons=c.mc.photons_per_batch; p.Nsims=1; p.NtimePts=c.mc.max_hops*hop_factor; p.useGpu=1;
if strcmp(variant,'legacy')
    launch=@legacy_mc.f_launch; hop=@legacy_mc.f_hop;
    spin=@legacy_mc.f_spin; back=@legacy_mc.f_backProp;
    % Historical f_simPSFs3D did not forward g/nt/nm. Retain defaults here.
else
    p.g=c.optics.g; p.nt=c.optics.nt; p.nm=c.optics.nm;
    launch=@f_launch; hop=@f_hop; spin=@f_spin; back=@f_backProp;
end
sum_kernel=zeros(p.Nx,p.Nx,'double'); escaped=0; collected=0; late=0; completed=0;
partial=[file '.partial.mat'];
if isfile(partial)
    s=load(partial); sum_kernel=s.sum_kernel; escaped=s.escaped;
    collected=s.collected; late=s.late; completed=s.completed;
end
bank_offset=0; if strcmp(bank,'test'), bank_offset=100000000; end
depth_offset=round(abs(z0)*1e4);
for batch=completed+1:c.mc.batches
    % Same random-stream seed for old/new. Test kernels use independent streams.
    seed=mod(c.mc.seed+bank_offset+depth_offset+plane*1000+batch,2^32-1);
    rng(seed,'twister'); gpurng(seed,'Threefry');
    [x,y,z,ux,uy,uz,L,atSurf]=launch(p);
    seen=false(p.Nphotons,1,'gpuArray'); late_batch=0;
    for j=1:p.NtimePts
        [x,y,z,L,atSurf]=hop(x,y,z,ux,uy,uz,L,p);
        if j>floor(p.NtimePts/2), late_batch=late_batch+gather(sum(~seen(atSurf))); end
        seen(atSurf)=true;
        [ux,uy,uz]=spin(ux,uy,uz,atSurf,p);
    end
    x=gather(x(atSurf)); y=gather(y(atSurf)); z=gather(z(atSurf));
    ux=gather(ux(atSurf)); uy=gather(uy(atSurf)); uz=gather(uz(atSurf));
    if isempty(x)
        one=zeros(p.Nx); xb=[];
    else
        [xb,~,~,one]=back(x,y,z,ux,uy,uz,p);
    end
    assert(all(isfinite(one(:))),'Nonfinite Monte Carlo kernel');
    sum_kernel=sum_kernel+one;
    escaped=escaped+numel(x); collected=collected+numel(xb); late=late+late_batch;
    completed=batch;
    if mod(batch,8)==0 || batch==c.mc.batches
        tmp=[partial '.tmp.mat'];
        save(tmp,'sum_kernel','escaped','collected','late','completed','-v7.3'); movefile(tmp,partial,'f');
    end
    fprintf('PSF %s %s %s plane=%d batch=%d/%d hops=%d\n',tag,variant,bank,plane,batch,c.mc.batches);
end
% hist3 Ctrs puts out-of-window counts in edge bins. Remove those bins exactly
% as f_simPSFs3D does; never renormalize away collection/FOV losses.
kernel=single(sum_kernel(2:end-1,2:end-1)/c.mc.batches);
total=c.mc.batches*c.mc.photons_per_batch;
diagnostics=struct('z0_um',z0,'variant',variant,'bank',bank,'photons',total,...
    'max_hops',p.NtimePts,'escaped_fraction',escaped/total,'collected_fraction',collected/total,...
    'in_fov_fraction',sum(kernel(:)),'late_escape_fraction_of_launched',late/total,...
    'g_used',p.g,'nt_used',p.nt,'nm_used',p.nm,'seed_formula','seed+bank_offset+round(abs(z0)*1e4)+plane*1000+batch');
final_tmp=[file '.complete.tmp.mat'];
save(final_tmp,'kernel','diagnostics','-v7.3'); movefile(final_tmp,file,'f');
if isfile(partial), delete(partial); end
fid=fopen([file '.json'],'w'); fprintf(fid,'%s\n',jsonencode(diagnostics)); fclose(fid);
end
