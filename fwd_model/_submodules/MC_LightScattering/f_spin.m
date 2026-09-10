
function [ux, uy, uz] = f_spin(ux,uy,uz,atSurf,pram)

  validateattributes(pram.g, {'numeric'}, ...
      {'real','scalar','finite','>=',-1,'<=',1}, mfilename, 'g');

  if pram.useGpu == 0
    rnd1      = rand(pram.Nphotons,1);
    rnd2      = rand(pram.Nphotons,1);
  else
    rnd1      = rand(pram.Nphotons,1,"gpuArray");
    rnd2      = rand(pram.Nphotons,1,"gpuArray");
  end

  if pram.g == 0
    costheta = 2*rnd1-1;  % isotropic scattering: avoid division by zero
  elseif abs(pram.g) == 1
    costheta = pram.g + zeros(size(rnd1), 'like', rnd1);
  elseif abs(pram.g) < 1e-3
    % Algebraically equivalent HG inverse CDF without cancellation at g=0.
    q = 2*rnd1-1;
    g = pram.g;
    costheta = (2*q + g*(q.^2+3) + 2*g^2*q + g^3*(q.^2-1)) ...
               ./ (2*(1+g*q).^2);
  else
    temp = (1-pram.g^2)./(1-pram.g+2*pram.g*rnd1);
    costheta = (1+pram.g^2-temp.^2)/(2*pram.g);
  end
  costheta  = min(1,max(-1,costheta));  % guard floating-point roundoff
  sintheta  = sqrt(max(0,1-costheta.^2));
  psi       = rnd2 * 2*pi;
     
  % **** for the following calculation see, Jacques_mcfluor2003.pdf page 32 ****
  % For unit directions this is sqrt(1-uz^2), but it stays accurate near
  % the poles. The exactly axial case needs a separate local basis.
  transverse = hypot(ux,uy);
  axial      = transverse == 0;
  temp       = transverse;
  temp(axial)= 1;  % safe placeholder; axial results are replaced below
  uxx       =  sintheta .* (ux .* uz .* cos(psi) - uy .* sin(psi)) ./ temp + ux .* costheta;
  uyy       =  sintheta .* (uy .* uz .* cos(psi) + ux .* sin(psi)) ./ temp + uy .* costheta;
  uzz       = -sintheta .* cos(psi) .* transverse                          + uz .* costheta;

  uxx(axial) = sintheta(axial).*cos(psi(axial));
  uyy(axial) = sintheta(axial).*sin(psi(axial));
  uzz(axial) = sign(uz(axial)).*costheta(axial);

  % Keep direction cosines normalized after repeated scattering events.
  unorm = sqrt(uxx.^2+uyy.^2+uzz.^2);
  uxx = uxx./unorm;
  uyy = uyy./unorm;
  uzz = uzz./unorm;

  uxx(atSurf) = ux(atSurf);           % replace the random direction with the original direction for atSurf photons
  uyy(atSurf) = uy(atSurf);
  uzz(atSurf) = uz(atSurf);
  
  ux          = uxx;
  uy          = uyy;
  uz          = uzz;    

end
