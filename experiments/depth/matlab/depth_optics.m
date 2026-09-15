function PSFs = depth_optics(c, pram)
% Shared optical PSFs: same equations/grids as upstream Efficient_PSF,
% evaluated in small row blocks without retaining three full complex volumes.
% This experiment changes scattering transport, not the optical Debye model.
cache = fullfile(c.run_dir,'psfs','optics.mat');
if isfile(cache)
    s = load(cache,'PSFs'); PSFs = s.PSFs; return
end
mkdir_if_needed(fileparts(cache));
dx = pram.dx / c.optics.sampling_factor;
nx = 2*c.optics.sampling_factor*pram.Nx + 1;
nz = c.optics.axial_planes;
ex = depth_optical_intensity(c.optics, c.optics.lambda_ex_um, dx, nx, nz, true).^2;
ex = ex / max(squeeze(sum(sum(ex,1),2)));
profile = squeeze(sum(sum(ex,1),2));
half = nz/2+1-find(profile>0.01,1,'first');
zrange = nz/2+1-half:nz/2+1+half;
assert(all(zrange>=1 & zrange<=nz),'Optical axial support exceeds configured grid; increase axial_planes');
PSFs.exPSF = ex(:,:,zrange); clear ex
em = depth_optical_intensity(c.optics, c.optics.lambda_em_um, dx, nx, nz, true);
em = em / max(squeeze(sum(sum(em,1),2)));
PSFs.emPSF = em(:,:,zrange); clear em
PSFs.pram.dx = dx;
PSFs.z_offsets_um = (-half:half)*dx;
PSFs.description = 'Shared upstream vectorial Debye equations; two-photon excitation intensity squared';
temp = [cache '.partial.mat']; save(temp,'PSFs','-v7.3'); movefile(temp,cache,'f');
end

function mkdir_if_needed(p)
if ~isfolder(p), mkdir(p); end
end
