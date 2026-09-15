function P = depth_prepare_forward(P,E,p)
% Reuse resized PSFs and blurred patterns for every object in this condition.
s=[P.pram.dx/p.dx,P.pram.dx/p.dx,P.pram.dx/p.dz];
P.ex_work=gpuArray(single(imresize3(P.exPSF,'Scale',s,'Method','linear','Antialiasing',true)));
P.em_work=gpuArray(single(imresize3(P.emPSF,'Scale',s,'Method','linear','Antialiasing',true)));
P.sc_work=gpuArray(single(imresize3(P.sPSF,'Scale',s,'Method','linear','Antialiasing',true)));
P.pattern_work=zeros([size(P.ex_work) p.Nt],'like',P.ex_work);
for j=1:p.Nt
    P.pattern_work(:,:,:,j)=f_conv3nd(P.ex_work,gpuArray(E(:,:,j)),'same');
end
end
