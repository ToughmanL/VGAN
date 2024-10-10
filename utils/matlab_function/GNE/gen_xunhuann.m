
clear,clc

wav_path = 'C:\Users\Lushangjun\Desktop\vowel_all_0912\data_vowel\Patient\S_00022_M';
dirOutput = dir(fullfile(wav_path,'*.wav'));

readfile = strings(length(dirOutput), 1);
for i = 1:length(dirOutput)
    readfile(i) = [dirOutput(i).folder, '\', dirOutput(i).name];
end


%������ַ����readfile
% readfile=[
%           'm11.wav';
%           'm36.wav';
% ];


%���readfile������
sizereadfile = size(readfile,1);
length_data = zeros(1,sizereadfile);

GNE_SEO_SNR_MEAN = zeros(1,sizereadfile); 
GNE_TKEO_SNR_MEAN= zeros(1,sizereadfile); 
GNE_SEO_NSR_MEAN=  zeros(1,sizereadfile); 
GNE_TKEO_NSR_MEAN= zeros(1,sizereadfile); 

GNE_MEAN =  zeros(1,sizereadfile);
GNE_STD  =  zeros(1,sizereadfile);
  
for NUMNUM = 1:sizereadfile
%     if NUMNUM == 97
    [y,fs] = audioread(readfile(NUMNUM,:));
    %length_data��ÿһ����������Ӧ�ĵ�������
    length_data(NUMNUM)=size(y,1);
    A = 0.3;
    B = 0.7;
    NBEGIN = ceil(A*length_data(NUMNUM));                             
    NEND   = ceil(B*length_data(NUMNUM));                                             
    [x,Fs] = audioread(readfile(NUMNUM,:),[NBEGIN,NEND]);   
    %===========================================================
    %================Ƕ���㷨BEGIN=============================

    % Downsample to 10kHz
    %============================================
    fs_caiyang = 10000;%����һ���²�����Ƶ��fs_caiyang
    %============================================
    if Fs> fs_caiyang
      Fsd = fs_caiyang;
      ratio = Fsd/Fs;
      s = resample(x,round(ratio*1000),1000); 
    %   if vflag; Tind = ceil(Tind*Fsd/Fs); end
    else
      s = x;
      Fsd = fs_caiyang;
    end

    % Inverse filter to obtain residual and work with residual instead
    win_length = round(30e-3*Fsd);		%Analysis window of 30 msecs.
    win_step   = round(10e-3*Fsd);		%Shift of 10 msecs. between frames
    P = 10;					%LPC order
    warning off;
    [ar,e,k] = lpcauto(s,P,[win_step win_length]);
    warning on;
    are = ar.*(sqrt(e)*ones(1,P+1));
    u = lpcifilt(s,ar,k(:,1));


    [F0,strength,Tind] = getF0(s,Fsd); 

    vind = find(F0 ~= 0);
    numframes = length(F0);
    GNE = zeros(1,numframes);

    GNE_SEO_SNR     = zeros(1,(length(vind)));
    GNE_TKEO_SNR    = zeros(1,(length(vind)));
    GNE_SEO_NSR     = zeros(1,(length(vind)));
    GNE_TKEO_NSR    = zeros(1,(length(vind)));

    for n=1:length(vind)
        framenum = vind(n);
        to = Tind(framenum,1);
        tn = Tind(framenum,2);
        analysis_region = u(to:tn); 


        %--�ڴ˼���������Ժ���500HZΪ�����Ƶ����
        NUM = ((fs_caiyang/2)/500) - 1;
        %-----------------------------------
        x_filter       = zeros(size(analysis_region,1),NUM);
        x_filter_h     = zeros(size(analysis_region,1),1);
        x_filter_l     = zeros(size(analysis_region,1),NUM-1);
        x_filter_seo   = zeros(size(analysis_region,1),NUM); 
        x_filter_seo_mean = zeros(1,size(analysis_region,1));
        x_filter_tkeo  = zeros(size(analysis_region,1)-2,NUM);
        %-----------------------------------
        x_filter_log       = zeros(size(analysis_region,1),NUM);
        x_filter_log_seo   = zeros(size(analysis_region,1),NUM);
        x_filter_log_seo_mean = zeros(1,size(analysis_region,1));
        x_filter_log_tkeo  = zeros(size(analysis_region,1)-2,NUM);
        %-----------------------------------               
        x_filter_tkeo_mean = zeros(1,size(analysis_region,1));      
        x_filter_log_tkeo_mean = zeros(1,size(analysis_region,1));  
        % ���õ�ͨ�˲����˳�ԭ�ź�0-500HZ��Ƶ�β�����洢��x_filter_h
        % Ȼ������SEO��TKEO�ĺ��������Ӧ��x_filter_seo_l��x_filter_tkeo_h
        h = filter_lowpass(450,550,0.1,52,Fsd);
        x_filter_h =filter(h,[1],analysis_region);
        %���ô�ͨ�˲����õ�500-4500HZ����500HZΪ��������˲��õ�������
        for i = 1:(NUM-1)
            wpl = 500*i;
            wph = 500*(i+1);
            wsl = 500*i - 100;
            wsh = 500*(i+1) - 100;
            [b,a] = filter_daitong(wpl,wph,wsl,wsh,Fsd);
            %���˲��Ժ��ÿһ��Ƶ�ε����ݶ���Ϊx_filter��һ��������
            x_filter_l(:,i)  = filter(b,a,analysis_region);
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
        %��ǰN����Ϊ�źţ�ʣ�µ���Ϊ����
        %VERF(SNR) = (u6+u7+u8+...u23)/(u1+u2+...uN)
        %VERF(NSR) = (u1+u2+...uN)/(u6+u7+u8+...u23)
        %============================================
        N = 5;      %������������趨ǰN��Ƶ���������źţ�ʣ�µ�Ƶ���������ź�
        %============================================
        SEO_SNR1 = 0;
        SEO_SNR2 = 0;
        TKEO_SNR1 = 0;
        TKEO_SNR2 = 0;

        SEO_NSR1 = 0;
        SEO_NSR2 = 0;
        TKEO_NSR1 = 0;
        TKEO_NSR2 = 0;
        for i = 1:N
            SEO_SNR1   = SEO_SNR1  +  x_filter_seo_mean(i);
            TKEO_SNR1  = TKEO_SNR1 +  x_filter_tkeo_mean(i);

            SEO_NSR2   = SEO_NSR2  +  x_filter_log_seo_mean(i);
            TKEO_NSR2  = TKEO_NSR2 +  x_filter_log_tkeo_mean(i);
        end
        for i=(N+1):(size(x_filter_seo_mean,2))
            SEO_SNR2   = SEO_SNR2  +  x_filter_seo_mean(i);
            TKEO_SNR2  = TKEO_SNR2 +  x_filter_tkeo_mean(i);

            SEO_NSR1   = SEO_NSR1  +  x_filter_log_seo_mean(i);
            TKEO_NSR1  = TKEO_NSR1 +  x_filter_log_tkeo_mean(i);
        end
        GNE_SEO_SNR(n)  = SEO_SNR1/SEO_SNR2;
        GNE_TKEO_SNR(n) = TKEO_SNR1/TKEO_SNR2;

        GNE_SEO_NSR(n)  = SEO_NSR1/SEO_NSR2;
        GNE_TKEO_NSR(n) = TKEO_NSR1/TKEO_NSR2;
    end

    GNE_SEO_SNR_MEAN(NUMNUM)   =  mean(GNE_SEO_SNR); 
    GNE_TKEO_SNR_MEAN(NUMNUM)  =  mean(GNE_TKEO_SNR);
    GNE_SEO_NSR_MEAN(NUMNUM)   =  mean(GNE_SEO_NSR);
    GNE_TKEO_NSR_MEAN(NUMNUM)  =  mean(GNE_TKEO_NSR);

    GNE_SEO_SNR_MEAN_ZHUAN  =  GNE_SEO_SNR_MEAN'; 
    GNE_TKEO_SNR_MEAN_ZHUAN =  GNE_TKEO_SNR_MEAN'; 
    GNE_SEO_NSR_MEAN_ZHUAN  =  GNE_SEO_NSR_MEAN';  
    GNE_TKEO_NSR_MEAN_ZHUAN =  GNE_TKEO_NSR_MEAN'; 
  
    %==========================================================
    %���¼���GNE�ľ�ֵ�ͱ�׼��
    GNE_MEAN(NUMNUM) =   mean(gne(x,fs));
    GNE_STD(NUMNUM)  =   std(gne(x,fs));

    GNE_MEAN_ZHUAN  =  GNE_MEAN';
    GNE_STD_ZHUAN   =  GNE_STD';
    %================Ƕ���㷨END===============================
    %===========================================================
%     end
end
% ��������� GNE_SEO_SNR_MEAN, GNE_TKEO_SNR_MEAN, GNE_MEAN
% (GNE_SEO_SNR_MEAN_ZHUAN, GNE_TKEO_SNR_MEAN_ZHUAN, GNE_MEAN_ZHUAN)

name = strings(length(readfile), 1);
for i = 1:length(dirOutput)
    temp_name = strsplit(readfile(i), '\');
    name(i) = temp_name(9);
end
data = [name, GNE_SEO_SNR_MEAN_ZHUAN, GNE_TKEO_SNR_MEAN_ZHUAN, GNE_MEAN_ZHUAN];
data(isnan(GNE_SEO_SNR_MEAN_ZHUAN), :) = [];
[m, n] = size(data);            
data_cell = mat2cell(data, ones(m,1), ones(n,1));    % ��data�и��m*n��cell����
title = {'filename', 'GNE_SEO_SNR', 'GNE_TKEO_SNR', 'GNE_MEAN'};
result = [title; data_cell];
split = strsplit(wav_path, '\');
xlswrite([wav_path, '\', char(split(7)), '_GNE.xlsx'], result);
fprintf("Done\n");
