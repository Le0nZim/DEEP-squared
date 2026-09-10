% 2020-xx-xx by Dushan N. Wadduwage (wadduwage@fas.harvard.edu)
% 2021-04-09 edited by dnw to include refraction effect and reejection of out of NA photons from sPSF

function [x_backProp, y_backProp, z_backProp, sPSF, sPSF_axis] = f_backProp(x,y,z,ux,uy,uz,pram)

  %% Refraction at a planar tissue/medium interface (normal is +z).
  % Snell: nt*sin(alpha1) = nm*sin(alpha2), with alpha measured from +z.
  % uz/|u| is COS(alpha1); the transverse magnitude gives SIN(alpha1).
  validateattributes(pram.nt, {'numeric'}, {'real','scalar','finite','positive'});
  validateattributes(pram.nm, {'numeric'}, {'real','scalar','finite','positive'});
  validateattributes(pram.NA, {'numeric'}, ...
      {'real','scalar','finite','nonnegative','<=',pram.nm});
  unorm         = sqrt(ux.^2+uy.^2+uz.^2);
  ux_out        = (pram.nt/pram.nm) .* ux./unorm;
  uy_out        = (pram.nt/pram.nm) .* uy./unorm;
  sinAlpha2_sq  = ux_out.^2+uy_out.^2;
  % A critical-angle ray travels along the surface and cannot reach the
  % objective. Retain only upward, propagating transmitted rays.
  refracted     = find(uz>0 & isfinite(unorm) & unorm>0 & sinAlpha2_sq<1);
  
  ux            = ux_out(refracted);
  uy            = uy_out(refracted);
  uz            = sqrt(1-sinAlpha2_sq(refracted));
  x             = x(refracted);
  y             = y(refracted);
  z             = z(refracted);
  
  %% filter out the photons out side the objective NA [2021-04-09]
  sinAlpha      = hypot(ux,uy);  % refracted directions are unit vectors
  inNA          = find(sinAlpha <= pram.NA/pram.nm);                        % by the definition of NA => NA = nm * sinAlpha_NA    
  
  ux            = ux(inNA);
  uy            = uy(inNA);
  uz            = uz(inNA);
  x             = x (inNA);
  y             = y (inNA);
  z             = z (inNA);
  
  %% [2020-xx-xx]
  s_backProp  = (pram.z0_um-z)./uz;
  
  x_backProp  = x + s_backProp .* ux;
  y_backProp  = y + s_backProp .* uy;
  z_backProp  = z + s_backProp .* uz;

  d_bin       = pram.dx;
  sPSF_axis   = (-floor(pram.Nx/2):floor(pram.Nx/2))*d_bin;
  if isempty(x_backProp)
    N = zeros(numel(sPSF_axis));
  else
    N = hist3(cat(2,x_backProp,y_backProp), ...
              'Ctrs',{sPSF_axis,sPSF_axis});
  end
                                                         
%  sPSF       = N/sum(N(:));                    % this is normalization to all escaped and in-range photons
  sPSF        = N/(pram.Nphotons*pram.Nsims);   % this is normalizaiton to all simulated photons. So NA effect is in here.
end
