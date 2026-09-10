

function [x, y, z, L, atSurf] = f_hop(x,y,z,ux,uy,uz,L,pram)

  if pram.useGpu == 0
    rnd1    = rand(pram.Nphotons,1);
  else
    rnd1    = rand(pram.Nphotons,1,"gpuArray");
  end

  s         = 1e4 * (-log(rnd1)/pram.mus);  % [um]   Step size. log() is base e. In the theory s = -ln(1-rnd)/μ_s [cm]... 
                                            %        But for any 2 random numbers rnd2 = 1-rnd1.
    
  atSurf    = find(uz > 0 & z + s .* uz >= 0); % only upward rays can exit
  s(atSurf) = max(0,-z(atSurf)./uz(atSurf)); % stop at the boundary, never hop backward
                                            %        boundary (i.e. a -z./uz distance)
  
  x         = x + s .* ux;
  y         = y + s .* uy;
  z         = z + s .* uz;
  z(atSurf) = 0;                          % exact boundary prevents roundoff drift on later hops

  L         = L + s;

end
