function [idx_pipe, row] = SearchPipeIdx(dict_pipe, col, value)

    [num_pipe, ~] = size(dict_pipe);

    for i = 1:num_pipe
        row_find = find(dict_pipe{i, col}(:) == value);

        if ~isempty(row_find)
            break;
        end

    end

    idx_pipe = i;
    row = row_find;
end
