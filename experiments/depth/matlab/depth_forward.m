function [Y0,gt] = depth_forward(X,E,PSFs,pram)
% Eq. 1 with the upstream convolution helpers, processing patterns sequentially.
% X is one specified object slab: no internal selection, random rotation, or scaling.
if ~isfield(PSFs,'ex_work'), PSFs=depth_prepare_forward(PSFs,E,pram); end
ex=PSFs.ex_work; em=PSFs.em_work; sc=PSFs.sc_work;
assert(size(X,3)==size(ex,3),'Object/PSF axial dimensions differ');
X=padarray(X,round([(size(ex,1)-pram.Ny)/2,(size(ex,2)-pram.Nx)/2,0]),0,'both');
X=X(1:size(ex,1),1:size(ex,2),:);
X=gpuArray(X); ex=gpuArray(ex); em=gpuArray(em); sc=gpuArray(sc);
yr=round(size(ex,1)/2-pram.Ny/2)+1:round(size(ex,1)/2+pram.Ny/2);
xr=round(size(ex,2)/2-pram.Nx/2)+1:round(size(ex,2)/2+pram.Nx/2);
zr=ceil(size(ex,3)/2); % Match f_fwd3D exactly, including even axial sizes.
target=f_conv3nd(ex,X,'same');
target=gather(target(yr,xr,zr));
assert(all(isfinite(target(:))) && max(target(:))>0,'Invalid/blank ground truth');
gt=single(target/max(target(:))); % one target shared across depth/PSF arms
Y0=zeros(pram.Ny,pram.Nx,pram.Nt,'single');
for pattern=1:pram.Nt
    excitation=PSFs.pattern_work(:,:,:,pattern);
    scattered=f_conv2nd(sc,excitation.*X,'same');
    detected=f_conv3nd(em,scattered,'same');
    Y0(:,:,pattern)=single(gather(detected(yr,xr,zr)));
end
assert(all(isfinite(Y0(:))) && all(Y0(:)>=0) && max(Y0(:))>0,'Invalid noiseless forward measurement');
end
