function hd=ideallp(wc,N);
tao=(N-1)/2;
n=[0:(N-1)];
m=n-tao+eps;
hd=sin(wc*m)./(pi*m);