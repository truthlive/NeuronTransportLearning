function WriteSWC(outtree, filename)
    %myFun - Description
    %
    % Syntax: output = WriteSWC(filename, Mat_swc)
    %
    % Long description
    fid = fopen(filename, 'w');
    fprintf(fid, '%s\n', '# TREES toolbox tree - job_checked_0001');
    fprintf(fid, '%s\n', '# written by an automatic procedure "swc_tree" part of the TREES package');
    fprintf(fid, '%s\n', '# in MATLAB');
    fprintf(fid, '%s\n', '#');
    fprintf(fid, '%s\n', '# copyright 2009 Hermann Cuntz');
    fprintf(fid, '%s\n', '# inode R X Y Z D / 2 idpar');

    [n_node, ~] = size(outtree)

    for i = 1:n_node
        fprintf(fid, '%d %d %12.8f %12.8f %12.8f %12.8f %d\n', outtree(i,:));
    end

    fclose(fid);

end
