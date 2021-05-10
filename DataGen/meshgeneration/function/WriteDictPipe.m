function WriteDictPipe(cell_pipe, filename)
    %myFun - Description
    %
    % Syntax: output = WriteSWC(filename, Mat_swc)
    %
    % Long description
    fid = fopen(filename, 'w');

    [num_pipe, ~] = size(cell_pipe);

    for idx_pipe = 1:num_pipe
        fprintf(fid, '%d %d %d ', [cell_pipe{idx_pipe, 1}(1), cell_pipe{idx_pipe, 2}(1), cell_pipe{idx_pipe, 3}(1)]);
        fprintf(fid, '%d ', cell_pipe{idx_pipe, end}(1:end));
        fprintf(fid, '\n');
    end

    fclose(fid);

end
