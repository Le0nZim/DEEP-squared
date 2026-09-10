function tests = test_scattering
% CPU regression tests for the production MATLAB transport functions.
% Run from the repository root:
%   results = runtests('fwd_model/tests/test_scattering.m');
%   assertSuccess(results);
% Requires the Statistics and Machine Learning Toolbox for hist3, as does
% the original f_backProp. No GPU, parallel pool, data files, or PSF run.
  tests = functiontests(localfunctions);
end

function setupOnce(testCase)
  testCase.TestData.oldPath = path;
  fwdDir = fileparts(fileparts(mfilename('fullpath')));
  addpath(fullfile(fwdDir,'_submodules','MC_LightScattering'),'-begin');
end

function teardownOnce(testCase)
  path(testCase.TestData.oldPath);
end

function setup(testCase)
  testCase.TestData.oldRng = rng;
  rng(20260910,'twister');
end

function teardown(testCase)
  rng(testCase.TestData.oldRng);
end

function p = parameters(n)
  p = struct('Nphotons',n,'Nsims',1,'useGpu',0, ...
      'g',0.9,'mus',200,'nt',1.33,'nm',1.33, ...
      'NA',1,'z0_um',-50,'Nx',21,'dx',1);
end

function testLaunchIsUniformInSolidAngle(testCase)
  p = parameters(200000);
  [~,~,~,ux,uy,uz] = f_launch(p);
  u = [ux,uy,uz];
  verifyLessThan(testCase,max(abs(mean(u,1))),0.006);
  verifyLessThan(testCase,max(abs(mean(u.^2,1)-1/3)),0.006);
  verifyLessThan(testCase,max(abs(sum(u.^2,2)-1)),1e-12);
  counts = histcounts(uz,linspace(-1,1,11));
  verifyLessThan(testCase,max(abs(counts/p.Nphotons-0.1)),0.005);
end

function testObjectiveAcceptsAxialCone(testCase)
  p = parameters(6);
  a = [0;10;40;48;50;80]*pi/180;
  tags = (1:6)';
  [~,yb] = f_backProp(zeros(6,1),tags,zeros(6,1), ...
      sin(a),zeros(6,1),cos(a),p);
  verifyEqual(testCase,yb,(1:4)');
end

function testBallisticCollectionMatchesSolidAngle(testCase)
  p = parameters(200000);
  [~,~,z,ux,uy,uz] = f_launch(p);
  up = uz>0;
  % An exact ballistic surface intercept, independent of hop and spin.
  s = -z(up)./uz(up);
  [xb,yb,zb,kernel] = f_backProp(s.*ux(up),s.*uy(up), ...
      zeros(sum(up),1),ux(up),uy(up),uz(up),p);
  expectedFraction = (1-sqrt(1-(p.NA/p.nm)^2))/2;
  verifyLessThan(testCase,abs(numel(xb)/p.Nphotons-expectedFraction),0.006);
  verifyLessThan(testCase,max(abs([xb;yb])),1e-10);
  verifyLessThan(testCase,max(abs(zb-p.z0_um)),1e-10);
  verifyEqual(testCase,sum(kernel(:)),numel(xb)/p.Nphotons,'AbsTol',1e-12);
  verifyEqual(testCase,nnz(kernel),1);
end

function testSnellRefractionAndTotalInternalReflection(testCase)
  p = parameters(3);
  p.nt = 1.5;
  p.nm = 1;
  p.NA = 1;
  a = [0;30;50]*pi/180;
  [xb,yb,zb] = f_backProp(zeros(3,1),(1:3)',zeros(3,1), ...
      sin(a),zeros(3,1),cos(a),p);
  % The 30-degree ray exits at asin(0.75); 50 degrees exceeds critical.
  verifyEqual(testCase,yb,[1;2]);
  verifyEqual(testCase,xb,[0;p.z0_um*0.75/sqrt(1-0.75^2)],'AbsTol',1e-11);
  verifyEqual(testCase,zb,[p.z0_um;p.z0_um],'AbsTol',1e-11);
end

function testRefractionBendsTowardNormalInHigherIndex(testCase)
  p = parameters(1);
  p.nt = 1;
  p.nm = 1.5;
  p.NA = 1.5;
  a = pi/3;
  [xb,~,~] = f_backProp(0,0,0,sin(a),0,cos(a),p);
  transmittedAngle = asin((p.nt/p.nm)*sin(a));
  verifyEqual(testCase,xb,p.z0_um*tan(transmittedAngle),'AbsTol',1e-11);
  verifyLessThan(testCase,abs(xb),abs(p.z0_um*tan(a)));
end

function testCollectionIndependentOfDirectionScale(testCase)
  p = parameters(2);
  a = [10;30]*pi/180;
  [x1,y1] = f_backProp([1;2],[3;4],[0;0],sin(a),[0;0],cos(a),p);
  [x2,y2] = f_backProp([1;2],[3;4],[0;0],3*sin(a),[0;0],3*cos(a),p);
  verifyEqual(testCase,x1,x2,'AbsTol',1e-11);
  verifyEqual(testCase,y1,y2,'AbsTol',1e-11);
end

function testEmptyCollectionAndDownwardRays(testCase)
  p = parameters(2);
  [xb,yb,zb,kernel,axis] = f_backProp([0;0],[0;0],[0;0], ...
      [0;1],[0;0],[-1;0],p);
  verifyEmpty(testCase,xb);
  verifyEmpty(testCase,yb);
  verifyEmpty(testCase,zb);
  verifySize(testCase,kernel,[21,21]);
  verifyEqual(testCase,sum(kernel(:)),0);
  verifyEqual(testCase,axis,-10:10);
end

function testHGScatteringMoments(testCase)
  p = parameters(200000);
  incoming = repmat([0.6,0,0.8],p.Nphotons,1);
  for g = [0,1e-10,-1e-10,0.9,-0.4]
    p.g = g;
    [ux,uy,uz] = f_spin(incoming(:,1),incoming(:,2),incoming(:,3),[],p);
    u = [ux,uy,uz];
    costheta = sum(u.*incoming,2);
    % HG Legendre moments: <P1(cos(theta))>=g, <P2(cos(theta))>=g^2.
    verifyLessThan(testCase,abs(mean(costheta)-g),0.006);
    verifyLessThan(testCase,abs(mean((3*costheta.^2-1)/2)-g^2),0.006);
    verifyLessThan(testCase,max(abs(sum(u.^2,2)-1)),1e-12);
    verifyTrue(testCase,all(isfinite(u(:))));
    verifyTrue(testCase,isreal(u));
  end
end

function testAxialAndNearlyAxialScattering(testCase)
  p = parameters(200000);
  for direction = [1,-1]
    for transverse = [0,1e-12]
      incoming = repmat([transverse,0,direction*sqrt(1-transverse^2)],p.Nphotons,1);
      [ux,uy,uz] = f_spin(incoming(:,1),incoming(:,2),incoming(:,3),[],p);
      u = [ux,uy,uz];
      verifyTrue(testCase,all(isfinite(u(:))));
      verifyTrue(testCase,isreal(u));
      verifyLessThan(testCase,max(abs(sum(u.^2,2)-1)),1e-12);
      verifyLessThan(testCase,abs(mean(sum(u.*incoming,2))-p.g),0.006);
    end
  end
end

function testPerfectForwardAndBackwardScattering(testCase)
  p = parameters(4);
  incoming = [0,0,1;0,0,-1;0.6,0,0.8;0,1,0];
  for g = [-1,1]
    p.g = g;
    [ux,uy,uz] = f_spin(incoming(:,1),incoming(:,2),incoming(:,3),[],p);
    verifyEqual(testCase,[ux,uy,uz],g*incoming,'AbsTol',1e-12);
  end
end

function testEscapedDirectionsStayFrozen(testCase)
  p = parameters(50);
  [~,~,~,ux,uy,uz] = f_launch(p);
  atSurf = (1:3:p.Nphotons)';
  [vx,vy,vz] = f_spin(ux,uy,uz,atSurf,p);
  verifyEqual(testCase,[vx(atSurf),vy(atSurf),vz(atSurf)], ...
      [ux(atSurf),uy(atSurf),uz(atSurf)]);
end

function testHopLengthsHaveCorrectUnitsAndDistribution(testCase)
  p = parameters(200000);
  v0 = zeros(p.Nphotons,1);
  [x,~,z,L,atSurf] = f_hop(v0,v0,p.z0_um+v0,ones(size(v0)),v0,v0,v0,p);
  meanFreePath = 1e4/p.mus;
  verifyLessThan(testCase,abs(mean(L)/meanFreePath-1),0.01);
  verifyLessThan(testCase,abs(mean(L<=meanFreePath)-(1-exp(-1))),0.006);
  verifyEqual(testCase,L,x);
  verifyEqual(testCase,z,p.z0_um+v0);
  verifyEmpty(testCase,atSurf);
end

function testBoundaryDoesNotMoveEscapedPhotons(testCase)
  p = parameters(2);
  x = [1;2]; y = [3;4]; z = [eps;0]; L = [7;8];
  ux = [0.6;-0.6]; uy = [0;0]; uz = [0.8;0.8];
  for k = 1:10
    [x,y,z,L,atSurf] = f_hop(x,y,z,ux,uy,uz,L,p);
    verifyEqual(testCase,x,[1;2]);
    verifyEqual(testCase,y,[3;4]);
    verifyEqual(testCase,z,[0;0]);
    verifyEqual(testCase,L,[7;8]);
    verifyEqual(testCase,atSurf,[1;2]);
  end
end

function testSmallTransportRunStaysFinite(testCase)
  p = parameters(2000);
  [x,y,z,ux,uy,uz,L] = f_launch(p);
  for k = 1:80
    [x,y,z,L,atSurf] = f_hop(x,y,z,ux,uy,uz,L,p);
    [ux,uy,uz] = f_spin(ux,uy,uz,atSurf,p);
  end
  state = [x,y,z,ux,uy,uz,L];
  verifyTrue(testCase,all(isfinite(state(:))));
  verifyTrue(testCase,isreal(state));
  verifyTrue(testCase,all(z<=0));
  verifyGreaterThan(testCase,numel(atSurf),0);
  [~,~,~,kernel] = f_backProp(x(atSurf),y(atSurf),z(atSurf), ...
      ux(atSurf),uy(atSurf),uz(atSurf),p);
  verifyTrue(testCase,all(isfinite(kernel(:))));
  verifyGreaterThanOrEqual(testCase,min(kernel(:)),0);
  verifyLessThanOrEqual(testCase,sum(kernel(:)),1);
end
