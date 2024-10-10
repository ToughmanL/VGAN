% function [FILE_s] = Gget_filelist(wDirectory, EXT)
%
% DESCRIPTION:
% ============
% get the list of audio files in a given folder
%
% INPUTS:
% =======
% - wDirectory	: full path to the folder
% - EXT			: extension of the files to look for (default [.wav])
%
% OUTPUTS:
% ========
% - FILE_s(:)	.root		: root of the audio file
%				.fullpath	: fullpath to the audio file
%
% Copyright (c) 2011 IRCAM/McGill, All Rights Reserved.
% Permission is only granted to use for research purposes
%

function [FILE_s] = Gget_filelist(wDirectory, EXT)

if nargin==1, EXT='.wav'; end

allfile_s = dir(wDirectory);
count=0;
for i=1:length(allfile_s)
    if isequal(allfile_s(i).name, '.') || isequal(allfile_s(i).name, '..') || ~allfile_s(i).isdir
        continue;
    end
    person_list = dir([allfile_s(i).folder, filesep, allfile_s(i).name]);
    for j = 1:length(person_list)
        if ~person_list(j).isdir
            if length(person_list(j).name)>length(EXT)
                if strcmp(person_list(j).name(end-length(EXT)+1:end), EXT)
                    count=count+1;
                    FILE_s(count).root		= [person_list(j).name(1:end-length(EXT))];
                    FILE_s(count).fullpath	= [person_list(j).folder filesep person_list(j).name];
                end % === if strcmp
            end % === if length
        end % === if isdir
    end
    
end % === for l

if ~exist('FILE_s'), FILE_s = []; end
