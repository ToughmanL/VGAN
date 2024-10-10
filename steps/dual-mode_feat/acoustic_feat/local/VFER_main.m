% 用此m脚本
clear,clc

addpath("utils/matlab_function/voicebox/");
addpath("utils/matlab_function/VFER/");
w_Directory = ['N_S_data/']; 
EXT = '.wav';
% 读取文件
[FILE_s] = Gget_filelist(w_Directory, EXT);

for num_file= 1:length(FILE_s)
    filename = [FILE_s(num_file).fullpath, '_VFER.txt'];
    if exist(filename,'file')
        continue
    end

    fprintf(1, '\nProcessing file %2d/%2d: %s\n', num_file, length(FILE_s), FILE_s(num_file).fullpath);
    fprintf(1, '=======================================\n');
    
    
    [y, fs] = audioread(FILE_s(num_file).fullpath);
    length_data = size(y,1);
    %length_data是每一个数据所对应的点数个数
    A = 0.1;
    B = 0.9;
    NBEGIN = ceil(A*length_data);
    NEND   = ceil(B*length_data);
    [s, f_s] = audioread(FILE_s(num_file).fullpath,[NBEGIN,NEND]);
    %
    %    [x,f_s1,bits1]=wavread(readfile(NUM,:),[ceil(C*length_data(NUM)),ceil(D*length_data(NUM))]);
    %===========================================================
    %================嵌入算法BEGIN=============================
    try
        VFER = vfer(s,f_s);
    catch
        continue
    end
    VFERmean = mean(VFER);
    VFERstd = std(VFER);
    
    %先把下采样后的s信号根据DYSPA声门张开和关闭的时间求取出每一个声门张开时候的点数
    
    
    %================嵌入算法END===============================
    %===========================================================
    % name_tmp = strsplit(FILE_s(num_file).fullpath, '\');
    % filename = char(name_tmp(end));
	% filename = [FILE_s(num_file).fullpath, '_VFER.txt'];
    data = [];
    data = [VFERmean, VFERstd];
    if ~isnan(VFERmean)
		fid		= fopen(filename, 'w');
		fprintf(fid, 'VFER_mean\tVFER_std\n');
		fprintf(fid, '%f\t%f\n', VFERmean, VFERstd);
        % [m, n] = size(data);
        % data_cell = [];
        % data_cell = mat2cell(data, ones(m,1), ones(n,1));    % 将data切割成m*n的cell矩阵
        % title = {'VFER_SEO_SNR', 'VFER_TKEO_SNR', 'VFER_mean', 'VFER_std'};
        % result = [];
        % result = [title; data_cell];
        % xlswrite([FILE_s(num_file).fullpath, '_VFER.xlsx'], result);
		fclose(fid);
    else
        fprintf("NaN: %s\n", filename);
    end
end
fprintf("Done\n");
