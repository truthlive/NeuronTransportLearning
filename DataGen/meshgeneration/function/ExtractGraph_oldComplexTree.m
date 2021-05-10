function [num_pipe, num_bif] = ExtractGraph_oldComplexTree(io_path)
    
    % Use different Bspline Smooth function to generate inserted nodes
    % along the skeleton of the complex tree

    parameter_file = [io_path, 'mesh_parameter.txt'];
    input_file = [io_path, 'skeleton_initial.swc'];
    smooth_file = [io_path, 'skeleton_smooth.swc'];
    pipe_path = [io_path, 'PipeSimulator//'];
    bif_path = [io_path, 'BifSimulator//'];

    output_dict_pipe = [io_path, 'dict_pipe.txt'];
    output_dict_bif = [io_path, 'dict_bif.txt'];
    output_graph_cnct = [io_path, 'GraphCnct.txt'];
    output_graph_file = [io_path, 'GraphRep.swc'];

    % Read skeleton information
    load_tree(input_file);
    trees{1} = load_tree(input_file);

    % Set smooth parameters
    var = LoadParameter(parameter_file);
    n_noisesmooth = var(1); % set iteration steps for noise smooth
    ratio_bifur_node = var(2); % set bifurcation nodes smooth ratio
    ratio_noisesmooth = var(3); % set noise smooth ratio
    seg_length = var(4); % set bezier smooth segments length
    ratio_refine = var(5); % set the ratio for the refinement around bifurcation point, not used in this code
    n_sample = var(6); % set the number of sample points to calculate the length of spline
    n_seg_per_chop = var(7); % set the number of segments in each short branch
    len_chop = var(8); % set largest length for the short branch

    % Set graph template
    % AllPipe = [];
    AllPipe = {};
    AllBif = []; % [branchpt, i, j, k]
    GraphCnct = [];
    output_graph_mat = [];

    vec_pipe2pipe = [1, 0, 0];
    vec_pipe2bif = [0.5, 0, 0];
    vec_bif2pipe = [0.5 * cos(pi / 3), 0.5 * sin(pi / 3), 0;
                0.5 * cos(pi / 3), -0.5 * sin(pi / 3), 0];

    %% Extract coordinates and connection information
    tangent_vec = [];
    location = [trees{1}.X, trees{1}.Y, trees{1}.Z];
    d = trees{1}.D;
    id = idpar_tree(trees{1});
    id = id';
    id(1) = 0;
    ipar = ipar_tree();

    branch1 = B_tree(trees{1});
    termination = T_tree(trees{1});
    [sect, vec] = dissect_tree(trees{1});
    [n_sect, ~] = size(sect);
    [n_bif, ~] = size(find(branch1 == 1));
    sect_point = cell(n_sect, 1);
    bif_term_pt = cell(n_sect, 1);
    bif_pt = cell(n_bif, 1);

    [nx, ny] = size(id);
    [m, n] = size(find(trees{1}.dA ~= 0));
    [mm, nn] = size(ipar);

    OutputData = [];

    %% Extract all points for each branch and all bifurcation points
    for i = 1:n_sect
        tmp_par = ipar(sect(i, 2), :);
        loc = find(tmp_par == sect(i, 1));
        sect_point{i} = tmp_par(loc:-1:1);
    end

    index_bif = 1;

    for sec_index = 1:n_sect
        [sect_ptnum, ~] = size(sect_point{sec_index}');
        start_pt_index = sect_point{sec_index}(1);
        end_pt_index = sect_point{sec_index}(sect_ptnum);

        bif_term_start = 1;
        bif_term_end = sect_ptnum;

        if (branch1(end_pt_index))
            par_node = id(end_pt_index);
            ans_tmp = find(trees{1}.dA(:, end_pt_index) == 1);
            bifur1_node = ans_tmp(1);
            bifur2_node = ans_tmp(2);
            bif_pt{index_bif} = [end_pt_index, par_node, bifur1_node, bifur2_node];
            index_bif = index_bif + 1;
            bif_term_end = sect_ptnum - 1;
        end

        if (branch1(start_pt_index))
            bif_term_start = 2;
        end

        bif_term_pt{sec_index} = sect_point{sec_index}(bif_term_start:bif_term_end);
    end

    %% Noise Smooth
    for index_smooth = 1:n_noisesmooth
        % Optimize bifurcation point to avoid bad geometry during mesh generation
        % Move the bifurcation points to the middle of three neighbour nodes
        for ii = 1:index_bif - 1
            pt_b = [trees{1}.X(bif_pt{ii}(1)), trees{1}.Y(bif_pt{ii}(1)), trees{1}.Z(bif_pt{ii}(1))]; d_b = trees{1}.D(bif_pt{ii}(1));
            pt_i = [trees{1}.X(bif_pt{ii}(2)), trees{1}.Y(bif_pt{ii}(2)), trees{1}.Z(bif_pt{ii}(2))]; d_i = trees{1}.D(bif_pt{ii}(2));
            pt_j = [trees{1}.X(bif_pt{ii}(3)), trees{1}.Y(bif_pt{ii}(3)), trees{1}.Z(bif_pt{ii}(3))]; d_j = trees{1}.D(bif_pt{ii}(3));
            pt_k = [trees{1}.X(bif_pt{ii}(4)), trees{1}.Y(bif_pt{ii}(4)), trees{1}.Z(bif_pt{ii}(4))]; d_k = trees{1}.D(bif_pt{ii}(4));
            pt_b_after = pt_b * (1 - ratio_bifur_node) + (pt_i + pt_j + pt_k) / 3 * ratio_bifur_node;
            d_b_after = d_b * (1 - ratio_bifur_node) + (d_i + d_j + d_k) / 3 * ratio_bifur_node;
            trees{1}.X(bif_pt{ii}(1)) = pt_b_after(1);
            trees{1}.Y(bif_pt{ii}(1)) = pt_b_after(2);
            trees{1}.Z(bif_pt{ii}(1)) = pt_b_after(3);
            trees{1}.D(bif_pt{ii}(1)) = d_b_after;
        end

        % Eliminate noise point
        for i = 1:n_sect
            AA = [trees{1}.X(sect_point{i}), trees{1}.Y(sect_point{i}), trees{1}.Z(sect_point{i}), trees{1}.D(sect_point{i})];
            BB = NoiseSmooth(AA, ratio_noisesmooth, 1);
            trees{1}.X(sect_point{i}) = BB(:, 1);
            trees{1}.Y(sect_point{i}) = BB(:, 2);
            trees{1}.Z(sect_point{i}) = BB(:, 3);
            trees{1}.D(sect_point{i}) = BB(:, 4);
        end

    end

    location = [trees{1}.X, trees{1}.Y, trees{1}.Z];
    d = trees{1}.D;

    inter_pt = [];

    for i = 1:n_sect
        inter_pt = [inter_pt, sect_point{i}(2:end - 1)];
    end

    trees{2} = delete_tree(trees{1}, inter_pt);
    branch2 = B_tree(trees{2});
    termination2 = T_tree(trees{2});

    [sect_after, vec_after] = dissect_tree(trees{2});
    location_after = [trees{2}.X, trees{2}.Y, trees{2}.Z];
    d_after = trees{2}.D;
    [ptnum, tmp] = size(d_after);
    branch_insert_vector = cell(ptnum, 3);

    %% Bezier Smooth

    num_pipe = 1;
    idx_graph = 1;

    for i = 1:n_sect

        start_index = sect_after(i, 1);
        end_index = sect_after(i, 2);

        if (branch2(end_index) && branch2(start_index))% pipe btw two bifs
            mode = 1;
            shift = 1;
        elseif (termination2(end_index))% pipe as terminal
            mode = 2;
            shift = 1;
        elseif (start_index == 1)% pipe as root
            mode = 3;
            shift = 0;

        end

        if n_bif == 0% single pipe
            mode = 4;
            shift = 0;
        end
        
        [tmp_XYZ,tmp_D, ~]=BsplineSmooth(location(sect_point{i},:),d(sect_point{i}),seg_length ,mode);

%         [tmp_XYZ, tmp_D, ~, tmp_n_seg] = BsplineSmoothEqualSeg(location(sect_point{i}, :), d(sect_point{i}), mode, n_sample, seg_length);
        [n_insert, ~] = size(tmp_D);
        % tangent_vec = [tangent_vec; start_index tmp_tangent(1, :)];

        i

        % * Categary the nodes in each chop

        % Group start node
        if (branch2(start_index))
            row_b = find(AllBif(:, 2) == start_index);
            [index, ~] = size(trees{2}.R);

            if AllBif(row_b, 4) == 0
                AllBif(row_b, 4) = index + 1;
            else
                AllBif(row_b, 5) = index + 1;
            end

            GraphCnct = [GraphCnct, [AllBif(row_b, 1); 1]];
        end

        if (start_index == 1)
            GraphCnct = [GraphCnct, [-1; 0]];
        end

        % Group internal node
        if mode == 1
%             tmp_pipe_idx = zeros(n_insert - 2, 1, 'int64');
            tmp_pipe_idx = zeros(n_insert - 2, 1);
            j_start = 1;
            j_end = n_insert - 2;
        elseif mode == 2
%             tmp_pipe_idx = zeros(n_insert - 1, 1, 'int64');
            tmp_pipe_idx = zeros(n_insert - 1, 1);
            tmp_pipe_idx(end) = sect_after(i, 2);
            j_start = 1;
            j_end = n_insert - 2;
        elseif mode == 3
%             tmp_pipe_idx = zeros(n_insert - 1, 1, 'int64');
            tmp_pipe_idx = zeros(n_insert - 1, 1);
            tmp_pipe_idx(1) = sect_after(i, 1);
            j_start = 2;
            j_end = n_insert - 1;
        elseif mode == 4
%             tmp_pipe_idx = zeros(n_insert, 1, 'int64');
            tmp_pipe_idx = zeros(n_insert, 1);
            tmp_pipe_idx(1) = sect_after(i, 1);
            tmp_pipe_idx(end) = sect_after(i, 2);
            j_start = 2;
            j_end = n_insert - 1;
        end

        [index, ~] = size(trees{2}.R);

        for j = j_start:j_end
            index = index + 1;
            tmp_pipe_idx(j) = index;
        end

        %         GraphCnct = [GraphCnct, [idx_graph; 1]];
        %         tmp_pipe_idx = [idx_graph; tmp_pipe_idx];

        AllPipe{end + 1} = idx_graph;
        AllPipe{end + 1} = tmp_pipe_idx(1);
        AllPipe{end + 1} = tmp_pipe_idx(end);
        AllPipe{end + 1} = tmp_pipe_idx;

        idx_graph = idx_graph + 1;

        % Group end node
        if (branch2(end_index))
            tmp_bifur_idx = [idx_graph, end_index, index, 0, 0];
            AllBif = [AllBif; tmp_bifur_idx];
            GraphCnct = [GraphCnct, [idx_graph - 1; 2]];
            idx_graph = idx_graph + 1;
        end

        % * Output single skeleton file for the whole branch without chopping into several short branches
        for j = 2:n_insert - 1
            [index, ~] = size(trees{2}.R);
            index = index + 1;
            % tangent_vec = [tangent_vec; index tmp_tangent(j, :)];

            if (j == 2)
                trees{2} = insert_tree(trees{2}, [index, 2, tmp_XYZ(j, :), tmp_D(j), sect_after(i, 1)]);
            else
                trees{2} = insert_tree(trees{2}, [index, 2, tmp_XYZ(j, :), tmp_D(j), index - 1]);
            end

        end

        trees{2} = recon_tree(trees{2}, sect_after(i, 2), index, 'none');

    end

    num_bif = n_bif;

    GraphCnct = GraphCnct';
    AllPipe = reshape(AllPipe, 4, []);
    AllPipe = AllPipe';

    [num_graphnode, ~] = size(GraphCnct);
    XYZ2 = [trees{2}.X, trees{2}.Y, trees{2}.Z];

    for i = 1:num_graphnode

        if GraphCnct(i, 2) == 0
            tmpXYZ = [0, 0, 0];
            output_graph_mat = [output_graph_mat; [i, 1, tmpXYZ, 0.1, GraphCnct(i, 1)]];

        elseif GraphCnct(i, 2) == 1
            row_p = find([AllPipe{:, 1}] == i);

            if GraphCnct(GraphCnct(i, 1), 2) == 2
                row_b = find(AllBif(:, 1) == GraphCnct(i, 1));
                tmpXYZ = (XYZ2(AllPipe{row_p, 2}(1), :) - XYZ2(AllBif(row_b, 2), :)) / ...
                    norm(XYZ2(AllPipe{row_p, 2}(1), :) - XYZ2(AllBif(row_b, 2), :));
            else
                tmpXYZ = (XYZ2(AllPipe{row_p, 2}(end), :) - XYZ2(AllPipe{row_p, 2}(1), :)) / ...
                    norm(XYZ2(AllPipe{row_p, 2}(end), :) - XYZ2(AllPipe{row_p, 2}(1), :));
            end

            tmpXYZ = tmpXYZ + output_graph_mat(GraphCnct(i, 1), 3:5);
            output_graph_mat = [output_graph_mat; [i, 1, tmpXYZ, 0.1, GraphCnct(i, 1)]];

        else
            row_b = find(AllBif(:, 1) == i);
            row_p = find([AllPipe{:, 1}] == GraphCnct(i, 1));
            tmpXYZ = (XYZ2(AllBif(row_b, 2), :) - XYZ2(AllPipe{row_p, 2}(end), :)) / ...
                norm(XYZ2(AllBif(row_b, 2), :) - XYZ2(AllPipe{row_p, 2}(end), :));
            tmpXYZ = tmpXYZ + output_graph_mat(GraphCnct(i, 1), 3:5);
            output_graph_mat = [output_graph_mat; [i, 2, tmpXYZ, 0.1, GraphCnct(i, 1)]];

        end

        i
        tmpXYZ
        %         output_graph_mat = [output_graph_mat;[i,2,tmpXYZ,0.1,GraphCnct(i,1)]];
    end

    WriteSWC(output_graph_mat, output_graph_file);

    %% Output pipe, bifur and graph

    WriteDictPipe(AllPipe, output_dict_pipe);
    % Table2_AllPipe = readtable(output_dict_pipe)
    % AllPipe2 = table2cell(Table2_AllPipe)

    % dlmwrite(output_dict_pipe, AllPipe, 'Delimiter', '\t');
    dlmwrite(output_dict_bif, AllBif, 'Delimiter', '\t');
    dlmwrite(output_graph_cnct, GraphCnct, 'Delimiter', '\t');

    %% Output and Visualization
    trees{1}.R(:, 1) = 2;
    trees{2}.R(:, 1) = 2;

    swc_tree(trees{2}, smooth_file);
    % fid4=fopen(tangent_file,'w');
    % [n_vec,tmp]=size(output_tangent);
    % for ii=1:n_vec
    %     fprintf(fid4,'%f %f %f\n',output_tangent(ii,2:4));
    % end
    fclose('all');

    figure(1); clf; xplore_tree(trees{1})
    figure(2); clf; xplore_tree(trees{2})

end
