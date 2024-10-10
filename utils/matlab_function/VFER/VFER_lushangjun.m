% 用此m文件
clear;clc

root_path = 'E:\Mono_Audio_only';

category = dir(root_path);
for categ_index = 3 : length(category)
    category_path = [category(categ_index).folder, '\', category(categ_index).name];
    person = dir(category_path);
    for person_index = 3 : length(person)
%         person(person_index).name
        if person(person_index).name == "S_00005_M" %|| person(person_index).name == "S_00033_F" || person(person_index).name == "S_00034_M" || person(person_index).name == "S_00035_M"
            wav_path = [person(person_index).folder, '\', person(person_index).name];
            dirOutput = dir(fullfile(wav_path,'*.wav'));
            
            readfile = strings(length(dirOutput), 1);
            for i = 1:length(dirOutput)
                readfile(i) = [dirOutput(i).folder, '\', dirOutput(i).name];
            end
            % readfile = [
            %             'm11.wav';
            %             'm36.wav'
            %             ];
            
            %求出readfile的行数
            sizereadfile = size(readfile,1);
            length_data =zeros(1,sizereadfile);
            
            VERF_SEO_SNR_MEAN = zeros(1,sizereadfile);
            VERF_TKEO_SNR_MEAN= zeros(1,sizereadfile);
            VERF_SEO_NSR_MEAN=  zeros(1,sizereadfile);
            VERF_TKEO_NSR_MEAN= zeros(1,sizereadfile);
            VFERmean = zeros(1,sizereadfile);
            VFERstd = zeros(1,sizereadfile);
            VFER_MEAN_ZHUAN = [];
            VFER_STD_ZHUAN = [];
            
            for NUM= 1:sizereadfile
                [y,fs] = audioread(readfile(NUM,:));
                %length_data是每一个数据所对应的点数个数
                length_data(NUM)=size(y,1);
                %    A = 0.3;
                %    B = 0.7;
                %    C = 0.1;
                %    D = 0.9;
                %    NBEGIN = ceil(A*length_data(NUM));
                %    NEND   = ceil(B*length_data(NUM));
                %    [s,f_s,bits]=wavread(readfile(NUM,:),[NBEGIN,NEND]);
                %
                %    [x,f_s1,bits1]=wavread(readfile(NUM,:),[ceil(C*length_data(NUM)),ceil(D*length_data(NUM))]);
                %===========================================================
                %================嵌入算法BEGIN=============================
                s=y;
                x=y;
                f_s=fs;
                f_s1=fs;
                
                VFER = vfer(x,f_s1);
                VFERmean(NUM) = mean(VFER)';
                VFERstd(NUM) = std(VFER)';
                
                VFER_MEAN_ZHUAN = VFERmean';
                VFER_STD_ZHUAN = VFERstd';
            end
            name = [];
            name = strings(length(readfile), 1);
            for i = 1:length(dirOutput)
                temp_name = strsplit(readfile(i), '\');
                name(i) = temp_name(end);   %
            end
            data = [];
            data = [name, VFER_MEAN_ZHUAN];
            data(isnan(VFER_MEAN_ZHUAN), :) = [];
            [m, n] = size(data);
            data_cell = [];
            data_cell = mat2cell(data, ones(m,1), ones(n,1));    % 将data切割成m*n的cell矩阵
            title = {'Filename', 'VFER_MEAN'};
            result = [];
            result = [title; data_cell];
            split = strsplit(wav_path, '\');
            xlswrite([wav_path, '\', char(split(end)), '_VFER.xlsx'], result);
            fprintf("Done: %s\n", person(person_index).name);
            
        end
        
        
    end
    
end

