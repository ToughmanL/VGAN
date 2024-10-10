% 用此m脚本
clear,clc
addpath("utils/matlab_function/voicebox/");
addpath("utils/matlab_function/GNE/");
w_Directory = ['N_S_data/'];
EXT = '.wav';

% 读取文件
[FILE_s] = Gget_filelist(w_Directory, EXT);

% ===   MAIN LOOP
for num_file = 1:length(FILE_s)
    filename = [FILE_s(num_file).fullpath, '_GNE.txt'];
    if exist(filename,'file')
        continue
    end

    fprintf(1, '\nProcessing file %2d/%2d: %s\n', num_file, length(FILE_s), FILE_s(num_file).fullpath);
    fprintf(1, '=======================================\n');
    
    [y,fs] = audioread(FILE_s(num_file).fullpath);
    %length_data是每一个数据所对应的点数个数
    length_data=size(y,1);
    A = 0.1;    %0.3
    B = 0.9;    %0.7
    NBEGIN = ceil(A*length_data);
    NEND   = ceil(B*length_data);
    [x,Fs] = audioread(FILE_s(num_file).fullpath,[NBEGIN,NEND]);
    %===========================================================
    %================嵌入算法BEGIN=============================
    
    % Downsample to 10kHz
    %============================================
    fs_caiyang = 10000;%设置一个下采样的频率fs_caiyang
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
    win_length = round(30e-3 * Fsd);		%Analysis window of 30 msecs.
    win_step   = round(10e-3 * Fsd);		%Shift of 10 msecs. between frames
    P = 10;					%LPC order
    warning off;
    if length(s) < win_length
        fprintf("wav length too short: %s\n", FILE_s(num_file).fullpath);
        GNE_SEO_SNR_MEAN   =  nan;
        GNE_TKEO_SNR_MEAN  =  nan;
        GNE_SEO_NSR_MEAN   =  nan;
        GNE_TKEO_NSR_MEAN  =  nan;
        GNE_MEAN =  nan;
        GNE_STD  =  nan;
    else
        
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
            try
                analysis_region = u(to:tn);
            catch
                continue;
            end
            
            %--在此计算出采样以后以500HZ为带宽的频段数
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
            % 利用低通滤波器滤出原信号0-500HZ的频段并将其存储在x_filter_h
            % 然后利用SEO和TKEO的函数求出相应的x_filter_seo_l和x_filter_tkeo_h
            h = filter_lowpass(450,550,0.1,52,Fsd);
            x_filter_h =filter(h,[1],analysis_region);
            % 利用带通滤波器得到500-4500HZ中以500HZ为带宽进行滤波得到的向量
            for i = 1:(NUM-1)
                wpl = 500*i;
                wph = 500*(i+1);
                wsl = 500*i - 100;
                wsh = 500*(i+1) - 100;
                [b,a] = filter_daitong(wpl,wph,wsl,wsh,Fsd);
                %将滤波以后的每一个频段的数据都存为x_filter的一个列向量
                x_filter_l(:,i)  = filter(b,a,analysis_region);
            end
            %将x_filter_l与x_filter_h合并到x_filter
            x_filter = [x_filter_h,x_filter_l];
            %对x_filter进行log变换得到x_filter_lo
            for i=1:size(x_filter,1)
                for j=1:size(x_filter,2)
                    if x_filter(i,j) > 0
                        x_filter_log(i,j) = log(x_filter(i,j));
                    else
                        x_filter_log(i,j) = 0;
                    end
                end
            end
            %分别对x_filter和x_filter_log求其SEO和TKEO,然后求其均值
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
            %下面就开始计算VERF的SNR和NSR
            %用前N个作为信号，剩下的作为噪声
            %VERF(SNR) = (u6+u7+u8+...u23)/(u1+u2+...uN)
            %VERF(NSR) = (u1+u2+...uN)/(u6+u7+u8+...u23)
            %============================================
            N = 5;      %这个参数用来设定前N个频段是有用信号，剩下的频段是噪声信号
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
        
        GNE_SEO_SNR_MEAN   =  mean(GNE_SEO_SNR);
        GNE_TKEO_SNR_MEAN  =  mean(GNE_TKEO_SNR);
        GNE_SEO_NSR_MEAN   =  mean(GNE_SEO_NSR);
        GNE_TKEO_NSR_MEAN  =  mean(GNE_TKEO_NSR);
        
        %==========================================================
        %以下计算GNE的均值和标准差
        try
            GNE_MEAN =   mean(gne(x,fs));
            GNE_STD  =   std(gne(x,fs));
        catch
            GNE_MEAN =  nan;
            GNE_STD  =  nan;
        end
    end
    
    %================嵌入算法END===============================
    %===========================================================
    %     end
    
    % 每个音频保存为一个特征
	% name_tmp = strsplit(FILE_s(num_file).fullpath, '\');
    % filename = char(name_tmp(end));
	% filename = [FILE_s(num_file).fullpath, '_GNE.txt'];
    data = [];
    data = [GNE_SEO_SNR_MEAN, GNE_TKEO_SNR_MEAN, GNE_MEAN, GNE_STD];
    if ~isnan(GNE_SEO_SNR_MEAN)
		fid		= fopen(filename, 'w');
		fprintf(fid, 'GNE_SEO_SNR\tGNE_TKEO_SNR\tGNE_MEAN\tGNE_STD\n');
		fprintf(fid, '%f\t%f\t%f\t%f\n', GNE_SEO_SNR_MEAN, GNE_TKEO_SNR_MEAN, GNE_MEAN, GNE_STD);
        % [m, n] = size(data);
        % data_cell = [];
        % data_cell = mat2cell(data, ones(m,1), ones(n,1));    % 将data切割成m*n的cell矩阵
        % title = {'GNE_SEO_SNR', 'GNE_TKEO_SNR', 'GNE_MEAN', 'GNE_STD'};
        % result = [];
        % result = [title; data_cell];
        %xlswrite([FILE_s(num_file).fullpath, '_GNE.xlsx'], result);
		fclose(fid);
    else
        fprintf("NaN: %s\n", filename);
    end 
end
fprintf("GNE done");