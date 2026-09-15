function depth_native_checks(c)
% Required, short native checks before a long production generation.
file=fullfile(c.run_dir,'native_checks.json');
if isfile(file), return; end
results=runtests(fullfile(c.repo_root,'fwd_model','tests','test_scattering.m'));
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
out=struct('matlab_release',version('-release'),'scattering_tests_passed',all([results.Passed]),...
    'optical_cpu_gpu_relative_max_error',relative,'experimental_data_loaded',false,...
    'note','Optical backend agreement is not an independent validation of Debye physics');
fid=fopen(file,'w'); fprintf(fid,'%s\n',jsonencode(out)); fclose(fid);
end
