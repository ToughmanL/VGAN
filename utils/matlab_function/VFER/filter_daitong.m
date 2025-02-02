function [b,a] =  filter_daitong(wpl,wph,wsl,wsh,f_s);
% f_s = 4.8e+04;
% wpl = 1000;
% wph = 1500;
% wsl = 900;
% wsh = 1400;
wpl = wpl * pi * 2 / f_s;
wph = wph * pi * 2 / f_s;
wsl = wsl * pi * 2 / f_s;
wsh = wsh * pi * 2 / f_s;
B = (wpl - wsl )/pi;
N = ceil(8 / B);
wc = [wpl/pi,wph/pi];
[b,a] = fir1(N-1,wc,hanning(N));
%[h,w] = freqz(b,a);
%h2 = abs(h);
%plot(w/(2 * pi) *f_s,h2);
%axis([500 2000 -0.2 1.1]);
