function [dict_pipe] = ReadDictPipe(filename)
    %myFun - Description
    %
    % Syntax: output = WriteSWC(filename, Mat_swc)
    %
    % Long description
    fid = fopen(filename, 'r');
    dict_pipe = {};

    while ~feof(fid)
        tline = fgetl(fid);
        tmp = str2num(tline);
        dict_pipe{end + 1} = tmp(1);
        dict_pipe{end + 1} = tmp(2);
        dict_pipe{end + 1} = tmp(3);
        dict_pipe{end + 1} = tmp(4:end)';
%         disp(tline)
    end
    fclose(fid);
    
    dict_pipe = reshape(dict_pipe, 4, []);
    dict_pipe = dict_pipe';
end
