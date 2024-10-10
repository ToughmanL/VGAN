clc;
clear;

%������ַ����readfile

wav_path = 'C:\Users\Lushangjun\Desktop\vowel_all_0912\Control\N_10003_M';
dirOutput = dir(fullfile(wav_path,'*.wav'));

readfile = strings(length(dirOutput), 1);
for i = 1:length(dirOutput)
    readfile(i) = [dirOutput(i).folder, '\', dirOutput(i).name];
end

%���readfile������
 sizereadfile = size(readfile,1);
 length_data =zeros(1,sizereadfile);
 
 VERF_SEO_SNR_MEAN = zeros(1,sizereadfile);
VERF_TKEO_SNR_MEAN= zeros(1,sizereadfile);
VERF_SEO_NSR_MEAN=  zeros(1,sizereadfile);
VERF_TKEO_NSR_MEAN= zeros(1,sizereadfile);

 for NUM= 1:3
  [y,fs] = audioread(readfile(NUM,:));
   %length_data��ÿһ����������Ӧ�ĵ�������
  length_data(NUM)=size(y,1);
   A = 0.3;
   B = 0.7;
   C = 0.1;
   D = 0.9;
   NBEGIN = ceil(A*length_data(NUM));                             
   NEND   = ceil(B*length_data(NUM));                                             
   [s,f_s]=audioread(readfile(NUM,:),[NBEGIN,NEND]);                              
   
   [x,f_s1]=audioread(readfile(NUM,:),[ceil(C*length_data(NUM)),ceil(D*length_data(NUM))]);                               
  %===========================================================
  %================Ƕ���㷨BEGIN=============================
    VFER = vfer(x,f_s1);
    VFERmean(NUM) = mean(VFER)';
    VFERstd(NUM) = std(VFER)';
  
    VFER_MEAN_ZHUAN = VFERmean';
    VFER_STD_ZHUAN = VFERstd';  
  
 
%�Ȱ��²������s�źŸ���DYSPA�����ſ��͹رյ�ʱ����ȡ��ÿһ�������ſ�ʱ��ĵ���
[gci,goi] = dypsa(s,f_s);
if goi(1) > gci(1)
    goi_new = [goi(1:size(goi,2)-1)];
    gci_new = [gci(2:size(gci,2))];
else 
    goi_new = goi;
    gci_new = gci;
end
openpoint = [goi_new;gci_new];
M = size(openpoint,2);%M�����ж��ٸ������ſ���ʱ���

VERF_SEO_SNR     = zeros(1,M);
VERF_TKEO_SNR    = zeros(1,M);
VERF_SEO_NSR     = zeros(1,M);
VERF_TKEO_NSR    = zeros(1,M);


for n = 1 : M
   x = s(openpoint(1,n) : openpoint(2,n)); 
%-----------------------------------
x_filter       = zeros(size(x,1),23);
x_filter_h     = zeros(size(x,1),1);
x_filter_l     = zeros(size(x,1),22);
x_filter_seo   = zeros(size(x,1),23);
x_filter_seo_mean = zeros(1,size(x,1));
x_filter_tkeo  = zeros(size(x,1)-2,23);
%-----------------------------------
x_filter_log       = zeros(size(x,1),23);
x_filter_log_seo   = zeros(size(x,1),23);
x_filter_log_seo_mean = zeros(1,size(x,1));
x_filter_log_tkeo  = zeros(size(x,1)-2,23);
%-----------------------------------
x_filter_tkeo_mean = zeros(1,size(x,1));
x_filter_log_tkeo_mean = zeros(1,size(x,1));
% ���õ�ͨ�˲����˳�ԭ�ź�0-500HZ��Ƶ�β�����洢��x_filter_h
% Ȼ������SEO��TKEO�ĺ��������Ӧ��x_filter_seo_l��x_filter_tkeo_h
h = filter_lowpass(450,550,0.1,52,f_s);
x_filter_h =filter(h,[1],x);
%���ô�ͨ�˲����õ�500-11500HZ����500HZΪ���������˲��õ�������
for i = 1:(11500-500)/500
    wpl = 500*i;
    wph = 500*(i+1);
    wsl = 500*i - 100;
    wsh = 500*(i+1) - 100;
    [b,a] = filter_daitong(wpl,wph,wsl,wsh,f_s);
    %���˲��Ժ��ÿһ��Ƶ�ε����ݶ���Ϊx_filter��һ��������
    x_filter_l(:,i)  = filter(b,a,x);
end
%��x_filter_l��x_filter_h�ϲ���x_filter
x_filter = [x_filter_h,x_filter_l];
%��x_filter����log�任�õ�x_filter_lo
for i=1:size(x_filter,1)
    for j=1:size(x_filter,2)
        if x_filter(i,j) > 0
            x_filter_log(i,j) = log(x_filter(i,j));
        else  x_filter_log(i,j) = 0;
        end
    end
end
%�ֱ��x_filter��x_filter_log����SEO��TKEO,Ȼ�������ֵ
for i=1:size(x_filter,2)
    x_filter_seo(:,i) = SEO(x_filter(:,i)');
    x_filter_seo_mean(i) = mean(x_filter_seo(:,i));
    x_filter_log_seo(:,i) = SEO(x_filter(:,i)');
    x_filter_log_seo_mean(i) = mean(x_filter_log_seo(:,i));
    %---------------------------------------
    x_filter_tkeo(:,i) = TKEO(x_filter(:,i)');
    x_filter_tkeo_mean(i) = mean(x_filter_tkeo(:,i));
    x_filter_log_tkeo(:,i) = TKEO(x_filter(:,i)');
    x_filter_log_tkeo_mean(i) = mean(x_filter_log_tkeo(:,i));
end
%����Ϳ�ʼ����VERF��SNR��NSR
%��ǰ�����Ϊ�źţ�ʣ�µ���Ϊ����
%VERF(SNR) = (u6+u7+u8+...u23)/(u1+u2+...u5)
%VERF(NSR) = (u1+u2+...u5)/(u6+u7+u8+...u23)
SEO_SNR1 = 0;
SEO_SNR2 = 0;
TKEO_SNR1 = 0;
TKEO_SNR2 = 0;

SEO_NSR1 = 0;
SEO_NSR2 = 0;
TKEO_NSR1 = 0;
TKEO_NSR2 = 0;
for i = 1:5
    SEO_SNR1   = SEO_SNR1  +  x_filter_seo_mean(i);
    TKEO_SNR1  = TKEO_SNR1 +  x_filter_tkeo_mean(i);
    
    SEO_NSR2   = SEO_NSR2  +  x_filter_log_seo_mean(i);
    TKEO_NSR2  = TKEO_NSR2 +  x_filter_log_tkeo_mean(i);
end
for i=6:(size(x_filter_seo_mean,2))
    SEO_SNR2   = SEO_SNR2  +  x_filter_seo_mean(i);
    TKEO_SNR2  = TKEO_SNR2 +  x_filter_tkeo_mean(i);
    
    SEO_NSR1   = SEO_NSR1  +  x_filter_log_seo_mean(i);
    TKEO_NSR1  = TKEO_NSR1 +  x_filter_log_tkeo_mean(i);
end




VERF_SEO_SNR(n)  = SEO_SNR1/SEO_SNR2;
VERF_TKEO_SNR(n) = TKEO_SNR1/TKEO_SNR2;

VERF_SEO_NSR(n)  = SEO_NSR1/SEO_NSR2;
VERF_TKEO_NSR(n) = TKEO_NSR1/TKEO_NSR2;
end



VERF_SEO_SNR_MEAN(NUM)   =  mean(VERF_SEO_SNR); 
VERF_TKEO_SNR_MEAN(NUM)  =  mean(VERF_TKEO_SNR);
VERF_SEO_NSR_MEAN(NUM)   =  mean(VERF_SEO_NSR);
VERF_TKEO_NSR_MEAN(NUM)  =  mean(VERF_TKEO_NSR);

   VERF_SEO_SNR_MEAN_ZHUAN  =  VERF_SEO_SNR_MEAN'; 
   VERF_TKEO_SNR_MEAN_ZHUAN =  VERF_TKEO_SNR_MEAN'; 
   VERF_SEO_NSR_MEAN_ZHUAN  =  VERF_SEO_NSR_MEAN';  
   VERF_TKEO_NSR_MEAN_ZHUAN =  VERF_TKEO_NSR_MEAN'; 
  
  %================Ƕ���㷨END===============================
  %===========================================================

  
 end

name = strings(length(readfile), 1);
for i = 1:length(dirOutput)
    temp_name = strsplit(readfile(i), '\');
    name(i) = temp_name(8);
end
data = [name, VERF_SEO_SNR_MEAN_ZHUAN, VERF_TKEO_SNR_MEAN_ZHUAN, VFER_MEAN_ZHUAN];
data(isnan(GNE_SEO_SNR_MEAN_ZHUAN), :) = [];
[m, n] = size(data);            
data_cell = mat2cell(data, ones(m,1), ones(n,1));    % ��data�и��m*n��cell����
title = {'filename', 'VFER_SEO_SNR', 'VFER_TKEO_SNR', 'VFER_MEAN'};
result = [title; data_cell];
split = strsplit(wav_path, '\');
xlswrite([wav_path, '\', char(split(7)), '_VFER.xlsx'], result);
fprintf("Done\n");














    
    
    
    
    
    
    
    
    