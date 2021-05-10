function [ var ] = LoadParameter( file_in )
% Load mesh generation parameter
    fileID = fopen(file_in);
    C=textscan(fileID,'%s %f');
    var = C{2};
    fclose(fileID);
end

