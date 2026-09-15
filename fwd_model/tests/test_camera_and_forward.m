function tests=test_camera_and_forward
% Native regression tests of production functions, not a Python noise proxy.
% Requires Statistics and Image Processing toolboxes; all tests here are CPU.
tests=functiontests(localfunctions);
end

function setupOnce(t)
t.TestData.old_path=path;
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'fwd_model'),fullfile(root,'experiments','depth','matlab'));
t.TestData.root=root;
t.TestData.temp=tempname; mkdir(t.TestData.temp);
end

function teardownOnce(t)
path(t.TestData.old_path);
rmdir(t.TestData.temp,'s');
end

function setup(t)
t.TestData.state=rng; rng(4125,'twister');
end

function teardown(t)
rng(t.TestData.state);
end

function p=parameters(t)
p=f_pram_init(); p.cam_N_gainStages=0; p.cam_Brnuli_alpha=0;
p.cam_EMgain=1; p.cam_sigma_rd=3; p.cam_dXdt_dark=0;
p.cam_ADCfactor=2; p.cam_bias=7; p.cam_t_exp=1;
p.emhist_dir=t.TestData.temp; p.emhist_trials=113;
p.dataset='unit_test'; p.repo_root=t.TestData.root;
end

function testReadNoiseIsIndependentAcrossPixelsPatternsAndSamples(t)
p=parameters(t);
[y,adu]=f_simulateIm_emCCD(zeros(2,3,32,4096),[],p);
verifySize(t,y,[2 3 32 4096]);
verifyEqual(t,adu,2*y+7,'AbsTol',1e-12);
verifyLessThan(t,abs(mean(y(:))),0.02);
verifyLessThan(t,abs(var(y(:))-9),0.1);
% Across independent batches, distinct pixels and distinct patterns must not
% be perfectly correlated. The old broadcast scalar fails these checks.
pairs=[squeeze(y(1,1,1,:)),squeeze(y(2,1,1,:)),squeeze(y(1,1,2,:))];
r=corrcoef(pairs); off=r(~eye(3));
verifyLessThan(t,max(abs(off)),0.06);
avg=mean(y,3);
verifyLessThan(t,abs(var(avg(:))-9/32),0.015);
verifyNotEqual(t,y(:,:,:,1),y(:,:,:,2));
end

function testLegacyActuallyHasBroadcastNoise(t)
p=parameters(t);
y=legacy_camera.f_simulateIm_emCCD(zeros(3,4,32,2),ones(1,1),p);
verifyEqual(t,y,repmat(y(1),size(y)));
end

function testOverflowDoesNotSaturate(t)
p=parameters(t); p.cam_sigma_rd=0;
table=struct('values',repmat((1:100)',1,17),'cam_N_gainStages',0,'cam_Brnuli_alpha',0);
y=f_simulateIm_emCCD(1000*ones(64,64),table,p);
verifyGreaterThan(t,min(y(:)),100);
verifyLessThan(t,abs(mean(y(:))-1000),3);
verifyGreaterThan(t,var(y(:)),800);
old=legacy_camera.f_simulateIm_emCCD(1000*ones(64,64),table.values,p);
verifyEqual(t,old,100*ones(64,64));
end

function testGainMismatchAndUnverifiedTablesAreRejected(t)
p=parameters(t);
table=struct('values',ones(10,11),'cam_N_gainStages',3,'cam_Brnuli_alpha',0.2);
verifyError(t,@() f_simulateIm_emCCD(ones(3),table,p),'DEEP2:EMGainMismatch');
verifyError(t,@() f_simulateIm_emCCD(ones(3),table.values,p),'DEEP2:UnverifiedEMTable');
verifyError(t,@() f_simulateIm_emCCD(-ones(3),[],p),'DEEP2:InvalidPhotonRate');
end

function testDirectRegisterHasExpectedMeanAndVariance(t)
p=parameters(t); p.cam_N_gainStages=5; p.cam_Brnuli_alpha=0.2;
p.cam_EMgain=1; p.cam_sigma_rd=0;
y=f_simulateIm_emCCD(4*ones(256,256),[],p);
gain=1.2^5;
% A Poisson input followed by independent Bernoulli branching at each stage.
conditional_variance=0.8/1.2*gain*(gain-1);
expected_variance=4*(gain^2+conditional_variance);
verifyLessThan(t,abs(mean(y(:))-4*gain),0.12);
verifyLessThan(t,abs(var(y(:))-expected_variance)/expected_variance,0.035);
end

function testArbitraryTableTrialCountsAndDeterminism(t)
p=parameters(t); p.cam_N_gainStages=3; p.cam_Brnuli_alpha=0.2;
for trials=[37 137]
    rng(71); a=f_genEmhist(5,trials,p);
    rng(71); b=f_genEmhist(5,trials,p);
    verifySize(t,a.values,[5 trials]); verifyEqual(t,a,b);
end
end

function testCameraCachePreservesRngAndExtendsWithoutChangingRows(t)
p=parameters(t); p.cam_N_gainStages=2; p.cam_Brnuli_alpha=0.2;
before=rng;
a=depth_emhist(p,3); verifyEqual(t,rng,before);
b=depth_emhist(p,7); verifyEqual(t,rng,before);
verifyEqual(t,b(1:3,:),a);
clear depth_emhist
verifyEqual(t,depth_emhist(p,7),b);
p.emhist_trials=47; small=depth_emhist(p,3);
verifySize(t,small,[3 47]);
end

function testLegacyBatchScopeSurvivesStreamingAndResume(t)
p=parameters(t); x=zeros(3,4,32);
a=depth_noise(x,p,1,'legacy',700);
b=depth_noise(x,p,2,'legacy',700);
verifyEqual(t,a,b); % Same generation group, different object/shot seeds.
c=depth_noise(x,p,3,'legacy',701); verifyNotEqual(t,c,a);
clear depth_noise
verifyEqual(t,depth_noise(x,p,2,'legacy',700),b);
fixed=depth_noise(x,p,1,'corrected',700);
verifyGreaterThan(t,var(double(fixed(:))),1);
verifyEqual(t,depth_noise(x,p,1,'corrected',700),fixed);
end

function testBlankAndLastValidObjectWindowsAndRectangularCrop(t)
p=parameters(t); p.useGPU=0; p.dx=1; p.dz=1;
p.Ny=6; p.Nx=8; p.Nt=2; p.dist=1; p.maxcount=20;
P.pram.dx=1; kernel=zeros(8,10,3,'single'); kernel(5,6,2)=1;
P.exPSF=kernel; P.emPSF=kernel; P.sPSF=kernel;
E=ones(6,8,2,'single');
[y,gt]=f_fwd3D(ones(6,8,3,'single'),E,P,[],p);
verifyEqual(t,size(y),[6 8 2]); verifyEqual(t,size(gt),[6 8]);
x=zeros(6,8,4,'single'); x(:,:,3)=1; x(:,:,4)=10;
[y,~]=f_fwd3D(x,E,P,[],p);
verifyEqual(t,size(y,4),1); % The final (second) valid slab is the only nonblank one.
[y,gt]=f_fwd3D(zeros(6,8,3,'single'),E,P,[],p);
verifyEmpty(t,y); verifyEmpty(t,gt);
end

function testOpticalRadiusAndAzimuthShareOneAxis(t)
o=struct('na',1,'nm',1.33,'theta_samples',40,'row_chunk',4);
for nx=[9 10]
    intensity=depth_optical_intensity(o,0.8,0.165,nx,8,false);
    center=floor(nx/2)+1; r=min(center-1,nx-center);
    image=intensity(center-r:center+r,center-r:center+r,5);
    verifyLessThan(t,max(abs(image(:)-reshape(flipud(image),[],1)))/max(image(:)),1e-5);
    verifyLessThan(t,max(abs(image(:)-reshape(fliplr(image),[],1)))/max(image(:)),1e-5);
end
end
