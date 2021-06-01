% Compute autocovariance function using FFT

function autocov = autocov(x, mean_x)
  x = x - mean_x;
  n = length(x);
  m = 1;
  while m<2*n
    m = 2*m;
  end
  f = fft(x,m);
  a = real(ifft(f.*conj(f),m));
  autocov = a(1:n);