function intensity = depth_optical_intensity(o, wavelength, dx, nx, nz, use_gpu)
% Memory-bounded evaluation of Efficient_PSF.m; preserve its sampling exactly.
alpha = asin(o.na/o.nm);
theta = (0:o.theta_samples-1) * alpha/o.theta_samples;
axis_xy = dx*(-nx/2:nx/2-1);
z = dx*(-nz/2:nz/2-1);
phi = calculate_phi(nx);
A = pi/wavelength;
intensity = zeros(nx,nx,nz,'single');
for first = 1:o.row_chunk:nx
    rows = first:min(first+o.row_chunk-1,nx);
    [X,Y,T] = meshgrid(axis_xy,axis_xy(rows),theta);
    V = (2*pi/wavelength)*sqrt(X.^2+Y.^2);
    F0 = sqrt(cos(T)).*sin(T).*(1+cos(T)).*besselj(0,V.*sin(T));
    F1 = sqrt(cos(T)).*sin(T).^2.*besselj(1,V.*sin(T));
    F2 = sqrt(cos(T)).*sin(T).*(1-cos(T)).*besselj(2,V.*sin(T));
    P = phi(rows,:);
    if use_gpu
        F0=gpuArray(F0); F1=gpuArray(F1); F2=gpuArray(F2); T=gpuArray(T); P=gpuArray(P);
    end
    for k = 1:nz
        phase = exp(-1i*(2*pi/wavelength)*z(k)*cos(T));
        I0=trapz(theta,F0.*phase,3); I1=trapz(theta,F1.*phase,3); I2=trapz(theta,F2.*phase,3);
        Ex=1i*A*(I0+I2.*cos(2*P)); Ey=1i*A*I2.*sin(2*P); Ez=-2*A*I1.*cos(P);
        value=abs(Ex).^2+abs(Ey).^2+abs(Ez).^2;
        if use_gpu, value=gather(value); end
        intensity(rows,:,k)=single(value);
    end
end
end
