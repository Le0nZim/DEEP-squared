function [PSFs, checks] = depth_psfs(c, pram, optical, variant, depth, bank)
tag=sprintf('%gsls',depth);
file=fullfile(c.run_dir,'psfs',tag,variant,[bank '_volume.mat']);
if isfile(file)
    s=load(file,'PSFs','checks'); PSFs=s.PSFs; checks=s.checks; return
end
PSFs=optical;
nz=numel(optical.z_offsets_um);
PSFs.sPSF=zeros(size(optical.emPSF),'single');
checks=cell(nz,1);
for plane=1:nz
    z0=-depth*1e4/c.optics.mus_cm_inv+optical.z_offsets_um(plane);
    assert(z0<0,'Scattering source crosses the tissue surface');
    [kernel,diag]=depth_mc(c,pram,variant,z0,bank,plane,tag,1);
    % Preserve the axial reversal in upstream f_simPSFs3D.
    PSFs.sPSF(:,:,nz-plane+1)=kernel;
    checks{plane}=diag;
    if c.mc.convergence_center_check && plane==ceil(nz/2) && strcmp(bank,'train')
        [longer,long_diag]=depth_mc(c,pram,variant,z0,bank,plane,tag,2);
        a=imresize(kernel,[16 16],'bilinear','Antialiasing',true);
        b=imresize(longer,[16 16],'bilinear','Antialiasing',true);
        shape_change=sum(abs(a(:)/max(sum(a(:)),eps)-b(:)/max(sum(b(:)),eps)));
        convergence=struct('base',diag,'double_hops',long_diag,...
            'relative_in_fov_mass_change',abs(sum(longer(:))-sum(kernel(:)))/max(sum(longer(:)),eps),...
            'coarse_shape_l1',shape_change);
        fid=fopen(fullfile(c.run_dir,'psfs',tag,variant,'convergence.json'),'w');
        fprintf(fid,'%s\n',jsonencode(convergence)); fclose(fid);
    end
end
temp=[file '.partial.mat']; save(temp,'PSFs','checks','-v7.3'); movefile(temp,file,'f');
end
