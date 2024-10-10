[x,fs,bits]=wavread('0001.wav',[10000,13999]);
X=fft(x);
fp=450;
fc=55;
Ap=0.1;
As=52;
wp=(fp/fs)*2*pi;
ws=(fc/fs)*2*pi;
tr_width=ws-wp;
M=ceil(6.6*pi/tr_width)+1;
wc=(wp+ws)/2;
n=[0:1:M-1];
hd=ideallp(wc,M);
w_ham=(hamming(M))';
h = hd .* w_ham;
%��h����fft�任
ffth = fft(h,1024);
w_new = (1:1:1024)/1024*2*pi*(fs/10);
%
[db,mag,pha,grd,w]=myfreqz(h,[1]);
dw =2*pi/1000;
Rp= - (min (db(1:1: wp/ dw +1)))%ʵ��ͨ������
As= - round (max(db(ws/dw +1: 1:501)))% ��С���˥��
[H,w]=freqz(h,[1],200);
dbmagH=20*log10(abs(H)/max(abs(H)));
angH=angle(H);

y1=filter(h,[1],x);
Y=fft(y1);
magY=abs(Y);
figure
subplot(211),plot(x);title('�˲�ǰ�����ź�');
subplot(212),plot(y1);title('�˲��������ź�');
figure
subplot(211),plot(w_new,abs(ffth));title('��Ƶ��˲���Ƶ��');



