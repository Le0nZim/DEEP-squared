
function [x, y, z, ux, uy, uz, L, atSurf] = f_launch(pram)

  x     = zeros(pram.Nphotons,1);                             % [um]      cartesian coordinates  
  y     = zeros(pram.Nphotons,1);
  z     = zeros(pram.Nphotons,1) + pram.z0_um;

  L     = zeros(pram.Nphotons,1);                             % [um]      path-length for each photon

  % An isotropic point source is uniform in solid angle, not in theta.
  % dOmega = dpsi * d(cos(theta)); see OMLC mc321.c, LAUNCH.
  uz       = 2*rand(pram.Nphotons,1)-1;
  sintheta = sqrt(max(0,1-uz.^2));
  psi      = rand(pram.Nphotons,1)*2*pi;

  ux    = sintheta.*cos(psi);                                 % propagation direction cosines
  uy    = sintheta.*sin(psi);

  if pram.useGpu == 1
    x   = gpuArray(x );
    y   = gpuArray(y );
    z   = gpuArray(z );
    ux  = gpuArray(ux);
    uy  = gpuArray(uy);
    uz  = gpuArray(uz);
    L   = gpuArray(L );    
  end
  
  atSurf = [];  
end
