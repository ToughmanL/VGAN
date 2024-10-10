% 用此m文件
clear;clc

root_path = 'E:\Mono_Audio_GNEVFER';

category = dir(root_path);

for categ_index = 3 : length(category)
    category_path = [category(categ_index).folder, '\', category(categ_index).name];
    if category(categ_index).name == "Control"
        person = dir(category_path);
        for person_index = 3 : length(person)
            if person(person_index).name == "N_10007_F"
                wav_path = [person(person_index).folder, '\', person(person_index).name];
                dirOutput = dir(fullfile(wav_path,'*.wav'));
                
                readfile = strings(length(dirOutput), 1);
                for i = 1:length(dirOutput)
                    readfile(i) = [dirOutput(i).folder, '\', dirOutput(i).name];
                end
                
                
                %求出readfile的行数
                sizereadfile = size(readfile,1);
                length_data =zeros(1,sizereadfile);
                
                VERF_SEO_SNR_MEAN = zeros(1,sizereadfile);
                VERF_SEO_SNR_MEAN_log = zeros(1,sizereadfile);
                VERF_TKEO_SNR_MEAN= zeros(1,sizereadfile);
                VERF_SEO_NSR_MEAN=  zeros(1,sizereadfile);
                VERF_TKEO_NSR_MEAN= zeros(1,sizereadfile);
                VFERmean = zeros(1,sizereadfile);
                VFERstd = zeros(1,sizereadfile);
                VFER_MEAN_ZHUAN = [];
                VFER_STD_ZHUAN = [];
                
                for NUM= 1:sizereadfile
                    [y,fs] = audioread(readfile(NUM,:));
                    NUM
%                     if NUM == 179
                        readfile(NUM,:)
                        %length_data是每一个数据所对应的点数个数
                        length_data(NUM)=size(y,1);
                        A = 0.2;
                        B = 0.8;
                        C = 0.1;
                        D = 0.9;
                        NBEGIN = ceil(A*length_data(NUM));
                        NEND   = ceil(B*length_data(NUM));
                        [s,f_s]=audioread(readfile(NUM,:),[NBEGIN,NEND]);
                        
                        [x,f_s1]=audioread(readfile(NUM,:),[ceil(C*length_data(NUM)),ceil(D*length_data(NUM))]);
                        %                 ===========================================================
                        %                 ================嵌入算法BEGIN=============================
                        
                        
                        VFER = vfer(x,f_s1);
                        VFERmean(NUM) = mean(VFER)';
                        VFERstd(NUM) = std(VFER)';
                        
                        VFER_MEAN_ZHUAN = VFERmean';
                        VFER_STD_ZHUAN = VFERstd';
                        
                        
                        %先把下采样后的s信号根据DYSPA声门张开和关闭的时间求取出每一个声门张开时候的点数
                        [gci,goi] = dypsa(s,f_s);
                        if goi(1) > gci(1)
                            goi_new = [goi(1:size(goi,2)-1)];
                            gci_new = [gci(2:size(gci,2))];
                        else
                            goi_new = goi;
                            gci_new = gci;
                        end
                        openpoint = [goi_new;gci_new];
                        M = size(openpoint,2);%M代表有多少个声门张开的时间段
                        
                        VERF_SEO_SNR     = zeros(1,M);
                        VERF_SEO_SNR_log = zeros(1,M);
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
                            % 利用低通滤波器滤出原信号0-500HZ的频段并将其存储在x_filter_h
                            % 然后利用SEO和TKEO的函数求出相应的x_filter_seo_l和x_filter_tkeo_h
                            h = filter_lowpass(450,550,0.1,52,f_s);
                            x_filter_h =filter(h,[1],x);
                            %利用带通滤波器得到500-11500HZ中以500HZ为带宽进行滤波得到的向量
                            for i = 1:(7500-500)/500
                                
                                wpl = 500*i;
                                wph = 500*(i+1);
                                wsl = 500*i - 100;
                                wsh = 500*(i+1) - 100;
                                [b,a] = filter_daitong(wpl,wph,wsl,wsh,f_s);
                                %将滤波以后的每一个频段的数据都存为x_filter的一个列向量
                                x_filter_l(:,i)  = filter(b,a,x);
                            end
                            %将x_filter_l与x_filter_h合并到x_filter
                            x_filter = [x_filter_h,x_filter_l];
                            %对x_filter进行log变换得到x_filter_lo
                            for i=1:size(x_filter,1)
                                for j=1:size(x_filter,2)
                                    if x_filter(i,j) > 0
                                        x_filter_log(i,j) = log(x_filter(i,j));
                                    else  x_filter_log(i,j) = 0;
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
                                %                         x_filter_tkeo(:,i) = TKEO(x_filter(:,i)');
                                %                         x_filter_tkeo_mean(i) = mean(x_filter_tkeo(:,i));
                                %                         x_filter_log_tkeo(:,i) = TKEO(x_filter(:,i)');
                                %                         x_filter_log_tkeo_mean(i) = mean(x_filter_log_tkeo(:,i));
                            end
                            %下面就开始计算VERF的SNR和NSR
                            %用前五个作为信号，剩下的作为噪声
                            %VERF(SNR) = (u6+u7+u8+...u23)/(u1+u2+...u5)
                            %VERF(NSR) = (u1+u2+...u5)/(u6+u7+u8+...u23)
                            SEO_SNR1 = 0; SEO_SNR1_log = 0;
                            SEO_SNR2 = 0; SEO_SNR2_log = 0;
                            TKEO_SNR1 = 0;
                            TKEO_SNR2 = 0;
                            
                            SEO_NSR1 = 0;
                            SEO_NSR2 = 0;
                            TKEO_NSR1 = 0;
                            TKEO_NSR2 = 0;
                            for i = 1:4
                                SEO_SNR1   = SEO_SNR1  +  x_filter_seo_mean(i);
                                SEO_SNR1_log = SEO_SNR1_log + x_filter_log_seo_mean(i);
                                %                         TKEO_SNR1  = TKEO_SNR1 +  x_filter_tkeo_mean(i);
                                %
                                %                         SEO_NSR2   = SEO_NSR2  +  x_filter_log_seo_mean(i);
                                %                         TKEO_NSR2  = TKEO_NSR2 +  x_filter_log_tkeo_mean(i);
                            end
                            for i=5:(size(x_filter_seo_mean,2))
                                SEO_SNR2   = SEO_SNR2  +  x_filter_seo_mean(i);
                                SEO_SNR2_log = SEO_SNR2_log + x_filter_log_seo_mean(i);
                                %                         TKEO_SNR2  = TKEO_SNR2 +  x_filter_tkeo_mean(i);
                                %
                                %                         SEO_NSR1   = SEO_NSR1  +  x_filter_log_seo_mean(i);
                                %                         TKEO_NSR1  = TKEO_NSR1 +  x_filter_log_tkeo_mean(i);
                            end
                            
                            VERF_SEO_SNR(n)  = SEO_SNR1/SEO_SNR2;
                            VERF_SEO_SNR_log(n)  = SEO_SNR1_log/SEO_SNR2_log;
                            %                     VERF_TKEO_SNR(n) = TKEO_SNR1/TKEO_SNR2;
                            %                     VERF_SEO_NSR(n)  = SEO_NSR1/SEO_NSR2;
                            %                     VERF_TKEO_NSR(n) = TKEO_NSR1/TKEO_NSR2;
                        end
                        
                        
                        VERF_SEO_SNR_MEAN(NUM)   =  mean(VERF_SEO_SNR);
                        VERF_SEO_SNR_MEAN_log(NUM)  =  mean(VERF_SEO_SNR_log);
                        %                 VERF_TKEO_SNR_MEAN(NUM)  =  mean(VERF_TKEO_SNR);
                        %                 VERF_SEO_NSR_MEAN(NUM)   =  mean(VERF_SEO_NSR);
                        %                 VERF_TKEO_NSR_MEAN(NUM)  =  mean(VERF_TKEO_NSR);
                        
                        VERF_SEO_SNR_MEAN_ZHUAN  =  VERF_SEO_SNR_MEAN';
                        VERF_SEO_SNR_MEAN_ZHUAN_log  =  VERF_SEO_SNR_MEAN_log';
                        %                 VERF_TKEO_SNR_MEAN_ZHUAN =  VERF_TKEO_SNR_MEAN';
                        %                 VERF_SEO_NSR_MEAN_ZHUAN  =  VERF_SEO_NSR_MEAN';
                        %                 VERF_TKEO_NSR_MEAN_ZHUAN =  VERF_TKEO_NSR_MEAN';
                        %================嵌入算法END===============================
                        %==========================================================
%                     end
                end
%                 name = [];
%                 name = strings(length(readfile), 1);
%                 for i = 1:length(dirOutput)
%                     temp_name = strsplit(readfile(i), '\');
%                     name(i) = temp_name(end);   %
%                 end
%                 data = [];
%                 data = [name, VFER_MEAN_ZHUAN, VERF_SEO_SNR_MEAN_ZHUAN, VERF_SEO_SNR_MEAN_ZHUAN_log];
%                 data(isnan(VFER_MEAN_ZHUAN), :) = [];
%                 [m, n] = size(data);
%                 data_cell = [];
%                 data_cell = mat2cell(data, ones(m,1), ones(n,1));    % 将data切割成m*n的cell矩阵
%                 title = {'Filename', 'VFER_MEAN', 'VFER_SEO', 'VFER_SEO_log'};
%                 result = [];
%                 result = [title; data_cell];
%                 split = strsplit(wav_path, '\');
%                 xlswrite([wav_path, '\', char(split(end)), '_VFER_2.xlsx'], result);
%                 fprintf("Done: %s\n", person(person_index).name);
                
            end
            
        end
    end
end

