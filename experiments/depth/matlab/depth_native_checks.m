function depth_native_checks(c)
% Required, short native checks before a long production generation.
file=fullfile(c.run_dir,'native_checks.json');
if isfile(file), return; end
results=runtests(fullfile(c.repo_root,'fwd_model','tests'));
assertSuccess(results);
% MATLAB/Python axis contract: HDF5 dimensions are reversed by h5py.
contract=fullfile(c.run_dir,'axis_contract.h5');
if isfile(contract), delete(contract); end
example=zeros(5,7,3,2,'single'); % MATLAB [Y,X,P,N]
example(2,6,3,2)=123;
h5create(contract,'/input',[7 5 3 2],'Datatype','single');
h5write(contract,'/input',permute(example,[2 1 3 4]));
% Independent CPU/GPU backend agreement for the shared optical implementation.
o=c.optics; o.theta_samples=40; o.row_chunk=4;
cpu=depth_optical_intensity(o,0.8,0.165,9,8,false);
gpu=depth_optical_intensity(o,0.8,0.165,9,8,true);
relative=max(abs(cpu(:)-gpu(:)))/max(cpu(:));
assert(relative<1e-4,'Shared optical CPU/GPU disagreement');
focal_error=check_even_focal_plane();
out=struct('matlab_release',version('-release'),'all_regression_tests_passed',all([results.Passed]),'regression_test_count',numel(results),...
    'forward_even_z_target_max_error',focal_error,'optical_cpu_gpu_relative_max_error',relative,'experimental_data_loaded',false,...
    'note','Optical backend agreement is not an independent validation of Debye physics');
fid=fopen(file,'w'); fprintf(fid,'%s\n',jsonencode(out)); fclose(fid);
end

function relative=check_even_focal_plane()
% Compare the actual old and new forward routines on an even axial support.
% The rectangular, centrosymmetric object is unchanged by old augmentation.
p=f_pram_init(); p.Nx=8; p.Ny=6; p.Nt=2; p.dx=1; p.dz=1;
p.dist=1; p.maxcount=20; p.useGPU=0;
p.cam_N_gainStages=0; p.cam_Brnuli_alpha=0; p.cam_EMgain=1;
p.cam_sigma_rd=0; p.cam_dXdt_dark=0;
P.pram.dx=1; P.exPSF=zeros(8,10,4,'single');
P.exPSF(5,6,:)=reshape(single([1 4 2 1]),1,1,4);
P.emPSF=zeros(8,10,4,'single'); P.emPSF(5,6,3)=1;
P.sPSF=zeros(8,10,4,'single'); P.sPSF(5,6,:)=1;
X=zeros(6,8,4,'single'); X(:,[2 7],1)=1; X([2 5],:,2)=2;
X(:,[3 6],3)=3; X([1 6],:,4)=4; E=ones(6,8,2,'single');
[~,upstream_gt]=f_fwd3D(X,E,P,[],p);
[~,runner_gt]=depth_forward(X,E,P,p);
relative=max(abs(double(runner_gt(:))-double(upstream_gt(:))/p.maxcount));
assert(relative<1e-5,'Even-Z focal-plane convention differs from f_fwd3D');
end
