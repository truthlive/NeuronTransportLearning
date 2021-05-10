function [] = GenHexGraph(io_path)
    %% User settings
    % Debug 3 bifurcation model

    skeleton_input = [io_path, 'skeleton_smooth.swc'];
    parameter_file = [io_path, 'mesh_parameter.txt'];
    velocity_output = [io_path, 'initial_velocityfield.txt'];

    input_pipe_dict = [io_path, 'dict_pipe.txt'];
    input_bif_dict = [io_path, 'dict_bif.txt'];
    input_graph_cnct = [io_path, 'GraphCnct.txt'];
    input_graph_file = [io_path, 'GraphRep.swc'];
    % pipe_path = [io_path, 'PipeSimulator//'];
    % bif_path = [io_path, 'BifSimulator//'];
    sim_path = [io_path, 'Simulator//'];

    % if ~exist(pipe_path, 'dir')
    %    mkdir(pipe_path)
    % end
    % if ~exist(bif_path, 'dir')
    %    mkdir(bif_path)
    % end
    if ~exist(sim_path, 'dir')
        mkdir(sim_path)
    end

    hex_output = [io_path, 'controlmesh.vtk'];
    hex_info_output = [io_path, 'controlmesh_info.txt'];
    hex_extract_output = [io_path, '//mesh_extract.vtk'];
    mapping_extract_output = [io_path, '//C2Dmapping_extract.txt'];
    mapping_extractvtk_output = [io_path, '//C2Dmapping_extract_forvtk.txt'];
    mapping_simulator_output = [io_path, '//mapping_simulator.txt'];

    % Read skeleton information
    trees{1} = load_tree(skeleton_input);

    % parameter setting
    var = LoadParameter(parameter_file);
    ratio_refine = var(5); % the parameter used to calculate the refinement around bifurcation region
    %% Extract skeleton information and initialize labels
    location = [trees{1}.X, trees{1}.Y, trees{1}.Z];
    d = trees{1}.D;
    id = idpar_tree(trees{1});
    id = id';
    id(1) = 0;
    ipar = ipar_tree(trees{1});

    branch = B_tree(trees{1});
    termination = T_tree(trees{1});

    [nx, ny] = size(id);
    [m, n] = size(find(trees{1}.dA ~= 0));
    [mm, nn] = size(ipar);

    in_point_label = -1;
    wall_label = 0;
    tip_label = 1;
    branch_label = -2;

    ijk_label = zeros(nx, 1); % i-0 j-1 k-2

    %% Read template mesh
    template_p = load('.//template//template_circle_points90.txt');
    % template_half_p=load('.//template//template_halfcircle_points90.txt');
    template_merge_p = load('.//template//template_merge120_points90.txt');

    template_e = load('.//template//template_circle_elements.txt');
    % template_half_e=load('.//template//template_halfcircle_elements.txt');
    template_merge_e = load('.//template//template_merge_elements.txt');
    template_merge_e_reverse = template_merge_e;

    [m1, ~] = size(template_p); %template points number
    [m2, ~] = size(template_e); %template elements number
    [m3, ~] = size(template_merge_p); %branch template points number
    [m4, ~] = size(template_merge_e); %branch template elements number

    for i = 1:m1
        velocity_value(i) = (1 - norm(template_p(i, :))^2);
    end

    inner_point_index = find(sqrt(template_p(:, 1).^2 + template_p(:, 2).^2 + template_p(:, 3).^2) < 0.95);
    boundary_point_index = find(sqrt(template_p(:, 1).^2 + template_p(:, 2).^2 + template_p(:, 3).^2) > 0.95);
    boundary_merge_point_index = find(sqrt(template_merge_p(:, 1).^2 + template_merge_p(:, 2).^2 + template_merge_p(:, 3).^2) > 0.95);

    for i = 1:m4

        if i > m4 / 3 * 2
            template_merge_e_reverse(i, :) = [template_merge_e(i, 1), template_merge_e(i, 2), template_merge_e(i, 5), template_merge_e(i, 4), template_merge_e(i, 3)];
        end

    end

    %% Construct bifurcation template
    tmp_merge_e_order = template_merge_e_reverse;
    % modify branch element order
    a = [90 91 92 93 95 96 97 100 101 105 125:134];
    b = [114 109 104 99 113 108 103 112 107 111 124:-2:116 123:-2:115];
    a = a + 1 + m2 / 2;
    b = b + 1 + m2 / 2;
    aa = a + m2 / 4;
    bb = b + m2 / 4;
    n = size(a, 2);

    for i = 1:n
        tmp_merge_e_order = swap(tmp_merge_e_order, a(i), b(i));
        tmp_merge_e_order = swap(tmp_merge_e_order, aa(i), bb(i));
    end

    for i = 1:m2 / 4
        tmp_merge_e_order = swap(tmp_merge_e_order, i + m2, i + m2 * 5/4);
    end

    % determine the boundary point order in template
    botbc_index = [];
    leftbc_index = [];
    rightbc_index = [];

    for i = 1:m3

        if (norm(template_merge_p(i, :)) > 0.95)

            if (template_merge_p(i, 2) > 0)
                leftbc_index = [leftbc_index; i];
            elseif (template_merge_p(i, 2) < 0)
                rightbc_index = [rightbc_index; i];
            end

            if (template_merge_p(i, 3) > 0)
                botbc_index = [botbc_index; i];
            end

        end

    end

    extra_pt = [2 25 124 217;
            4 32 125 218]; % extraordinary points

    % assign template element order to branch
    template_branch_bottom = template_e;
    template_branch_left = [template_e(1:m2 / 2, :); template_merge_e((m2 + 1):end, :)];
    % template_branch_right=[template_merge_e((m2+1):end,:);template_e(m4+1:end,:)];
    template_branch_right = [tmp_merge_e_order((m2 + 1):end, :); template_e(m2 / 2 + 1:end, :)];

    template_branch_pt_bot = template_merge_p(1:m1, :);
    template_branch_pt_left = template_merge_p(1:m1, :);
    template_branch_pt_right = template_merge_p(1:m1, :);
    template_branch_pt_bot_terminal = template_p;

    rotate90_cw120 = [1 0 0; 0 cosd(-120) -sind(-120); 0 sind(-120) cosd(-120)];
    rotate90_ccw120 = [1 0 0; 0 cosd(120) -sind(120); 0 sind(120) cosd(120)];

    for i = 1:m1

        if template_merge_p(i, 2) < 0
            template_branch_pt_left(i, :) = template_merge_p(i, :) * rotate90_ccw120;
        end

        if template_merge_p(i, 2) > 0
            template_branch_pt_right(i, :) = template_merge_p(i, :) * rotate90_cw120;
        end

    end

    %% * Set extracted points
    extract_p = load('.//template//extraction//template_circle_point90_extract_index.txt');
    extract_merge_p = load('.//template//extraction//template_merge120_points90_extract_index.txt');

%     extract_p_theta = load('.//template//extraction//template_circle_point90_extract_index_theta.txt');

    extract_p(:, 2) = extract_p(:, 2) + 1;
    extract_merge_p(:, 2) = extract_merge_p(:, 2) + 1;

    extract_e = load('.//template//extraction//template_circle_elements_extract_index.txt');
    extract_merge_e = load('.//template//extraction//template_merge_elements_extract_index.txt');

    extract_merge_e_bot = load('.//template//extraction//template_merge_elements_extract_index_bot.txt');
    extract_merge_e_left = load('.//template//extraction//template_merge_elements_extract_index_left.txt');
    extract_merge_e_right = load('.//template//extraction//template_merge_elements_extract_index_right.txt');

    % dict_pipe = load(input_pipe_dict);
    dict_pipe = ReadDictPipe(input_pipe_dict);
    dict_bif = load(input_bif_dict);

    [m1_extract, ~] = size(extract_p);
    [m2_extract, ~] = size(extract_e);
    [m3_extract, ~] = size(extract_merge_p);
    [m4_extract, ~] = size(extract_merge_e);

    % * Extraction variable
    Layer_extract = cell(ny, 4);
    LayerElement_extract = [];
    AllPoint_extract = [];
    AllPointFeature_extract = [];
    PointMapping_extract = [];
    PointMapping_simulator = [];
    PointLabel_simulator = [];
    AllElement_extract = [];
    AllElementLabel_extract = [];

    tmp_extract_p = extract_p;
    tmp_extract_mp = extract_merge_p;
    NodeLayer_extract = cell(ny, 5);
    BranchLayerPoint_extract = cell(ny, 3);

    %% Define the temporary point and element in each layer
    tmp_p = template_p;
    tmp_mp = template_merge_p;
    tmp_e = zeros(m2, 8);
    tmp_me = zeros(m4, 8);
    tmp_branch_be = zeros(m2, 8);
    tmp_branch_le = zeros(m2, 8);
    tmp_branch_re = zeros(m2, 8);
    n_layer = zeros(ny, ny);
    branch_element_location = cell(ny);
    Segment_Vector = cell(ny, ny);
    Tangent_Vector = cell(ny, 3);
    NodeLayer = cell(ny, 5);
    BranchLayerPoint = cell(ny, 3);

    AllPoint = [];
    LayerElement = [];
    AllElement = [];
    AllElementLabel = [];
    AllLabel = [];
    AllVelocity = [];
    AllLayerIndex = [];

    for i = 1:ny

        for j = 1:ny

            if trees{1}.dA(i, j) ~= 0
                Segment_Vector{i, j} = location(i, :) - location(j, :);
                % sv=location(i,:)-location(j,:);
                %             n_layer(i,j)=ceil(norm(Segment_Vector{i,j})/(d(j)*1.1));
                %             if(termination(i) || j==1)
                %                 n_layer(i,j)=ceil(norm(Segment_Vector{i,j})/(d(j)*0.25));
                %             end
                % set layer number for normal segments
                n_layer(i, j) = 1;
                % set layer number segments around bifurcation for refinement
                % the bifurcation diameter is used to calculate the total layer numbers
                if (branch(i))
                    n_layer(i, j) = ceil(norm(Segment_Vector{i, j}) / (d(i) * ratio_refine));
                    %                 n_layer(i, j) = 4;
                end

                if (branch(j))
                    n_layer(i, j) = ceil(norm(Segment_Vector{i, j}) / (d(j) * ratio_refine));
                    %                 n_layer(i, j) = 4;
                end

            end

        end

    end

    %% Skeleton segmentation
    % Group the all branches to two catergories:
    % -- 1. branch between two bifurcation points;
    % -- 2. branch with one bifurcation point and one termination end
    [sect, ~] = dissect_tree(trees{1});
    [n_sect, ~] = size(sect);
    [n_bif, ~] = size(find(branch == 1));
    sect_pt = cell(n_sect, 1); %save pt index for each section (may include branch node)
    bif_term_pt = cell(n_sect, 1); % save pt index for bif-term section (branch node excluded)
    % bif_pt=cell(n_bif,1); %save pt index for bif region,
    bif_pt = zeros(n_bif, 4);
    bif_ele = zeros(n_bif, 4); %save ele index for bif region,
    %order:[bifur_node, par_node, bifur1_node, bifur2_node]

    % * Extraction
    sect_pt_extract = cell(n_sect, 1); %save pt index for each section (may include branch node)
    bif_term_pt_extract = cell(n_sect, 1); % save pt index for bif-term section (branch node excluded)
    bif_pt_extract = zeros(n_bif, 4);
    bif_ele_extract = zeros(n_bif, 4); %save ele index for bif region,
    %order:[bifur_node, par_node, bifur1_node, bifur2_node]

    for i = 1:n_sect
        tmp_par = ipar(sect(i, 2), :);
        loc = find(tmp_par == sect(i, 1));
        sect_pt{i} = tmp_par(loc:-1:1);
    end

    index_bif = 1;

    for index_sec = 1:n_sect
        [sect_ptnum, ~] = size(sect_pt{index_sec}');
        index_start_pt = sect_pt{index_sec}(1);
        index_end_pt = sect_pt{index_sec}(sect_ptnum);

        bif_term_start = 1;
        bif_term_end = sect_ptnum;

        if (branch(index_end_pt))
            par_node = id(index_end_pt);
            ans_tmp = find(trees{1}.dA(:, index_end_pt) == 1);
            bifur1_node = ans_tmp(1);
            bifur2_node = ans_tmp(2);
            bif_pt(index_bif, :) = [index_end_pt, par_node, bifur1_node, bifur2_node];
            index_bif = index_bif + 1;
            bif_term_end = sect_ptnum - 1;
        end

        if (branch(index_start_pt))
            bif_term_start = 2;
        end

        bif_term_pt{index_sec} = sect_pt{index_sec}(bif_term_start:bif_term_end);
    end

    %% Generate layers for all bifurcation regions
    for index_bif = 1:n_bif
        i = bif_pt(index_bif, 1);
        par_node = bif_pt(index_bif, 2);
        bifur1_node = bif_pt(index_bif, 3);
        bifur2_node = bif_pt(index_bif, 4);
        sv_parpar = Segment_Vector{par_node, id(par_node)};
        sv_par = Segment_Vector{i, par_node}; ri = d(par_node) / 2.0;
        sv_bifur1 = Segment_Vector{bifur1_node, i}; rj = d(bifur1_node) / 2.0; %rj=(ri+rj)/2;
        sv_bifur2 = Segment_Vector{bifur2_node, i}; rk = d(bifur2_node) / 2.0; %rk=(ri+rk)/2;
        ijk_label(bifur1_node) = 1;
        ijk_label(bifur2_node) = 2;

        tmp_mp = template_merge_p * (ri + rj + rk) / 3;
        tmp_bp = template_branch_pt_bot * (ri + rj + rk) / 3;
        tmp_lp = template_branch_pt_left * (ri + rj + rk) / 3;
        tmp_rp = template_branch_pt_right * (ri + rj + rk) / 3;

        tmp_bpt = template_branch_pt_bot_terminal * ri;
        tmp_lpt = template_branch_pt_bot_terminal * rj;
        tmp_rpt = template_branch_pt_bot_terminal * rk;

        %Using local coordinates to calculate three half planes
        %Bifurcation point is origin ([0,0,0])
        vi = -sv_par / norm(sv_par);
        vj = sv_bifur1 / norm(sv_bifur1);
        vk = sv_bifur2 / norm(sv_bifur2);

        aij = acos(dot(vi, vj) / norm(vi) / norm(vj));
        aik = acos(dot(vi, vk) / norm(vi) / norm(vk));
        akj = acos(dot(vk, vj) / norm(vk) / norm(vj));

        Kij = (ri * vj + rj * vi) / norm((ri * vj + rj * vi));
        Kik = (ri * vk + rk * vi) / norm((ri * vk + rk * vi));
        Kkj = (rk * vj + rj * vk) / norm((rk * vj + rj * vk));

        if (dot(cross(vi, Kij), cross(vi, Kik)) > 0)

            if aij > aik
                Kij = -Kij;
            else
                Kik = -Kik;
            end

        end

        if (aij <= pi / 2)
            spij = Kij * ri / sin(atan(ri / rj));
        else
            spij = Kij * (ri + rj) / 2;
        end

        if (aik <= pi / 2)
            spik = Kik * ri / sin(atan(ri / rk));
        else
            spik = Kik * (ri + rk) / 2;
        end

        if (akj <= pi / 2)
            spkj = Kkj * rk / sin(atan(rk / rj));
        else
            spkj = Kkj * (rk + rj) / 2;
        end

        cpn = cross((spik - spkj), (spij - spkj)) / norm(cross((spik - spkj), (spij - spkj)));
        fprintf('i=%d original_cpn=[%f %f %f]\n', i, cpn(1), cpn(2), cpn(3));

        % Branch Template rotation
        cp1 = cpn * (ri + rj + rk) / 3;
        cp2 = -cpn * (ri + rj + rk) / 3;

        fprintf('i=%d cpn=[%f %f %f]\n', i, cpn(1), cpn(2), cpn(3));

        w_bpt = cross(sv_par, cp1); w_bpt = w_bpt / norm(w_bpt);
        w_lpt = cross(sv_bifur1, cp1); w_lpt = w_lpt / norm(w_lpt);
        w_rpt = cross(sv_bifur2, cp1); w_rpt = w_rpt / norm(w_rpt);

        for k = 1:m1
            tmp_bpt(k, :) = cp1 / norm(cp1) * tmp_bpt(k, 1) + w_bpt * tmp_bpt(k, 2);
            tmp_lpt(k, :) = cp1 / norm(cp1) * tmp_lpt(k, 1) + w_lpt * tmp_lpt(k, 2);
            tmp_rpt(k, :) = cp1 / norm(cp1) * tmp_rpt(k, 1) + w_rpt * tmp_rpt(k, 2);
        end

        %Deal with half plane separate points
        ciji = w_bpt * ri - sv_par;
        ciki = -w_bpt * ri - sv_par;
        cijj = w_lpt * rj + sv_bifur1;
        cjkj = -w_lpt * rj + sv_bifur1;
        cjkk = w_rpt * rk + sv_bifur2;
        cikk = -w_rpt * rk + sv_bifur2;

        ni_layer = n_layer(i, par_node);
        nj_layer = n_layer(bifur1_node, i);
        nk_layer = n_layer(bifur2_node, i);

        ciji = (spij * (ni_layer - 1) + ciji) / ni_layer;
        ciki = (spik * (ni_layer - 1) + ciki) / ni_layer;
        cijj = (spij * (nj_layer - 1) + cijj) / nj_layer;
        cjkj = (spkj * (nj_layer - 1) + cjkj) / nj_layer;
        cikk = (spik * (nk_layer - 1) + cikk) / nk_layer;
        cjkk = (spkj * (nk_layer - 1) + cjkk) / nk_layer;

        kij1 = sqrt(norm(ciji)^2 + norm(spij)^2 - 2 * dot(spij, ciji));
        kij2 = sqrt(norm(cijj)^2 + norm(spij)^2 - 2 * dot(spij, cijj));
        spij = (ciji * kij2 + cijj * kij1) / (kij1 + kij2);

        kik1 = sqrt(norm(ciki)^2 + norm(spik)^2 - 2 * dot(spik, ciki));
        kik2 = sqrt(norm(cikk)^2 + norm(spik)^2 - 2 * dot(spik, cikk));
        spik = (ciki * kik2 + cikk * kij1) / (kik1 + kik2);

        kjk1 = sqrt(norm(cjkj)^2 + norm(spkj)^2 - 2 * dot(spkj, cjkj));
        kjk2 = sqrt(norm(cjkk)^2 + norm(spkj)^2 - 2 * dot(spkj, cjkk));
        spkj = (cjkj * kjk2 + cjkk * kjk1) / (kjk1 + kjk2);

        for k = 1:m1

            if (template_branch_pt_bot(k, 2) < 0)
                tmp_bp(k, :) = cp1 * template_branch_pt_bot(k, 1) + spik * norm(template_branch_pt_bot(k, 2:3));
            elseif (template_branch_pt_bot(k, 2) > 0)
                tmp_bp(k, :) = cp1 * template_branch_pt_bot(k, 1) + spij * norm(template_branch_pt_bot(k, 2:3));
            else
                tmp_bp(k, :) = cp1 * template_branch_pt_bot(k, 1);
            end

            if (template_branch_pt_left(k, 3) < 0)
                tmp_lp(k, :) = cp1 * template_branch_pt_left(k, 1) + spij * norm(template_branch_pt_left(k, 2:3));
            elseif (template_branch_pt_left(k, 3) > 0)
                tmp_lp(k, :) = cp1 * template_branch_pt_left(k, 1) + spkj * template_branch_pt_left(k, 3);
            else
                tmp_lp(k, :) = cp1 * template_branch_pt_left(k, 1);
            end

            if (template_branch_pt_right(k, 3) < 0)
                tmp_rp(k, :) = cp1 * template_branch_pt_right(k, 1) + spik * norm(template_branch_pt_right(k, 2:3));
            elseif (template_branch_pt_right(k, 3) > 0)
                tmp_rp(k, :) = cp1 * template_branch_pt_right(k, 1) + spkj * template_branch_pt_right(k, 3);
            else
                tmp_rp(k, :) = cp1 * template_branch_pt_right(k, 1);
            end

        end

        for k = 1:m3

            if (template_merge_p(k, 3) > 0)%plane separate jk
                tmp_mp(k, :) = cp1 * template_merge_p(k, 1) + spkj * template_merge_p(k, 3);
            elseif template_merge_p(k, 2) > 0%plane separate ij
                tmp_mp(k, :) = cp1 * template_merge_p(k, 1) + spij * norm(template_merge_p(k, 2:3));
            elseif template_merge_p(k, 2) < 0%plane separate ik
                tmp_mp(k, :) = cp1 * template_merge_p(k, 1) + spik * norm(template_merge_p(k, 2:3));
            else
                tmp_mp(k, :) = cp1 * template_merge_p(k, 1);
            end

        end

        % Move points to branch location
        for inode = 1:m3
            tmp_mp(inode, :) = tmp_mp(inode, :) + location(i, :) - location(1, :);
        end

        for inode = 1:m1
            tmp_bp(inode, :) = tmp_bp(inode, :) + location(i, :) - location(1, :);
            tmp_lp(inode, :) = tmp_lp(inode, :) + location(i, :) - location(1, :);
            tmp_rp(inode, :) = tmp_rp(inode, :) + location(i, :) - location(1, :);
            tmp_bpt(inode, :) = tmp_bpt(inode, :) + location(i, :) - location(1, :) - sv_par;
            tmp_lpt(inode, :) = tmp_lpt(inode, :) + location(i, :) - location(1, :) + sv_bifur1;
            tmp_rpt(inode, :) = tmp_rpt(inode, :) + location(i, :) - location(1, :) + sv_bifur2;
        end

        % Save Branch points and Branch layer element
        tmp_label = ones(m3, 1);
        tmp_label = in_point_label * tmp_label;
        tmp_label(boundary_merge_point_index, :) = wall_label;

        for ii = 1:m3
            tmp_veloctity_branch(ii, 1:3) = [0 0 0];
        end

        [tmp_pointnumber, tmp] = size(AllPoint);
        bif_ele(index_bif, 1) = tmp_pointnumber; % record the element information around bifurcation
        AllPoint = [AllPoint; tmp_mp];
        AllLabel = [AllLabel; tmp_label];
        AllVelocity = [AllVelocity; tmp_veloctity_branch];

        for k = 1:m4
            tmp_me(k, 1:4) = template_merge_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
        end

        for k = 1:m2
            tmp_branch_be(k, 1:4) = template_branch_bottom(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
            tmp_branch_le(k, 1:4) = template_branch_left(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
            tmp_branch_re(k, 1:4) = template_branch_right(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
        end

        LayerElement = [LayerElement; tmp_me(:, 1:4)];
        NodeLayer{i, 1} = tmp_mp;
        NodeLayer{i, 2} = tmp_branch_be(:, 1:4);
        NodeLayer{i, 3} = tmp_branch_le(:, 1:4);
        NodeLayer{i, 4} = tmp_branch_re(:, 1:4);
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        NodeLayer{i, 5} = [dict_bif(idx_bif, 1), dict_bif(idx_bif, 1)];
        BranchLayerPoint{i, 1} = tmp_bp;
        BranchLayerPoint{i, 2} = tmp_lp;
        BranchLayerPoint{i, 3} = tmp_rp;

        tmp_label = ones(m1, 1);
        tmp_label = in_point_label * tmp_label;
        tmp_label(boundary_point_index, :) = wall_label;

        % * Save Branch points and Branch layer element (For extraction)
        [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
        bif_ele_extract(index_bif, 1) = tmp_pointnumber_extract; % record the element information around bifurcation
        AllPoint_extract = [AllPoint_extract; tmp_mp(extract_merge_p(:, 2), :)];
        PointMapping_extract = [PointMapping_extract; extract_merge_p(:, 2) + tmp_pointnumber];
        gidx_extract = [1:m3_extract] + tmp_pointnumber_extract;
        lidx1_extract = [1:m3_extract];
        lidx2_extract = [1:m3_extract];
        PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];
        % AllLabel=[AllLabel;tmp_label];
        % AllVelocity=[AllVelocity;tmp_veloctity_branch];

        for k = 1:m4_extract
            tmp_me_extract(k, 1:4) = extract_merge_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
        end

        for k = 1:m2_extract
            tmp_branch_be_extract(k, 1:4) = extract_merge_e_bot(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
            tmp_branch_le_extract(k, 1:4) = extract_merge_e_left(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
            tmp_branch_re_extract(k, 1:4) = extract_merge_e_right(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
        end

        LayerElement_extract = [LayerElement_extract; tmp_me_extract(:, 1:4)];
        NodeLayer_extract{i, 1} = tmp_mp(tmp_extract_mp(:, 2));
        NodeLayer_extract{i, 2} = tmp_branch_be_extract(:, 1:4);
        NodeLayer_extract{i, 3} = tmp_branch_le_extract(:, 1:4);
        NodeLayer_extract{i, 4} = tmp_branch_re_extract(:, 1:4);
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        NodeLayer_extract{i, 5} = [dict_bif(idx_bif, 1), dict_bif(idx_bif, 1)];
        PointLabel_simulator = [PointLabel_simulator; [ones(m3_extract, 1) * dict_bif(idx_bif, 1), ones(m3_extract, 1) * dict_bif(idx_bif, 1)]];

        BranchLayerPoint_extract{i, 1} = tmp_bp(tmp_extract_p(:, 2));
        BranchLayerPoint_extract{i, 2} = tmp_lp(tmp_extract_p(:, 2));
        BranchLayerPoint_extract{i, 3} = tmp_rp(tmp_extract_p(:, 2));

        % tmp_label=ones(m1,1);
        % tmp_label=in_point_label*tmp_label;
        % tmp_label(boundary_point_index,:)=wall_label;

        % Save bottom terminal points and element
        if (par_node == 1)
            tmp_label = ones(m1, 1);
            tmp_label = tip_label * tmp_label;
            tmp_label(boundary_point_index, :) = wall_label;
            tip_label = tip_label + 1;
        end

        for ii = 1:m1
            tmp_veloctity(ii, 1:3) = sv_parpar / norm(sv_parpar) * norm(velocity_value(ii));
        end

        [tmp_pointnumber, tmp] = size(AllPoint);
        AllPoint = [AllPoint; tmp_bpt];
        AllLabel = [AllLabel; tmp_label];
        AllVelocity = [AllVelocity; tmp_veloctity];

        for k = 1:m2
            tmp_e(k, 1:4) = template_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
        end

        LayerElement = [LayerElement; tmp_e(:, 1:4)];
        NodeLayer{par_node, 1} = tmp_bpt;
        NodeLayer{par_node, 2} = tmp_e(:, 1:4);
        NodeLayer{par_node, 3} = tmp_e(:, 1:4);
        NodeLayer{par_node, 4} = cp1;
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 3, par_node);
        NodeLayer{par_node, 5} = [dict_bif(idx_bif, 1), dict_pipe{idx_pipe, 1}];
        % record the element information around bifurcation
        bif_ele(index_bif, 2) = tmp_pointnumber;

        % * Save bottom terminal points and element  (For extraction)
        [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
        AllPoint_extract = [AllPoint_extract; tmp_bpt(extract_p(:, 2), :)];
        PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];
        gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;
        % [idx_pipe, col] = find(dict_pipe(:, 2:end) == par_node);
        [~, row] = SearchPipeIdx(dict_pipe, 4, par_node);
        % [row, ~] = size(dict_pipe{idx_pipe, 4});
        lidx1_extract = [1:m1_extract] + m1_extract * (row - 1);
        lidx2_extract = [1:m1_extract] + m3_extract;
        PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];

        for k = 1:m2_extract
            tmp_e_extract(k, 1:4) = extract_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
        end

        LayerElement_extract = [LayerElement_extract; tmp_e_extract(:, 1:4)];
        NodeLayer_extract{par_node, 1} = tmp_bpt(tmp_extract_p(:, 2));
        NodeLayer_extract{par_node, 2} = tmp_e_extract(:, 1:4);
        NodeLayer_extract{par_node, 3} = tmp_e_extract(:, 1:4);
        NodeLayer_extract{par_node, 4} = cp1;
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 3, par_node);
        NodeLayer_extract{par_node, 5} = [dict_bif(idx_bif, 1), dict_pipe{idx_pipe, 1}];
        % PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * dict_bif(idx_bif, 1), ones(m1_extract, 1) * dict_pipe{idx_pipe, 1}]];
        PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * dict_pipe{idx_pipe, 1}, ones(m1_extract, 1) * dict_bif(idx_bif, 1)]];

        % record the element information around bifurcation
        bif_ele_extract(index_bif, 2) = tmp_pointnumber_extract;

        % Save left terminal points and element
        if (termination(bifur1_node))
            tmp_label = ones(m1, 1);
            tmp_label = tip_label * tmp_label;
            tmp_label(boundary_point_index, :) = wall_label;
            tip_label = tip_label + 1;

        end

        for ii = 1:m1
            tmp_veloctity(ii, 1:3) = sv_bifur1 / norm(sv_bifur1) * norm(velocity_value(ii));
        end

        [tmp_pointnumber, ~] = size(AllPoint);
        AllPoint = [AllPoint; tmp_lpt];
        AllLabel = [AllLabel; tmp_label];
        AllVelocity = [AllVelocity; tmp_veloctity];

        for k = 1:m2
            tmp_e(k, 1:4) = template_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
        end

        LayerElement = [LayerElement; tmp_e(:, 1:4)];
        NodeLayer{bifur1_node, 1} = tmp_lpt;
        NodeLayer{bifur1_node, 2} = tmp_e(:, 1:4);
        NodeLayer{bifur1_node, 3} = tmp_e(:, 1:4);
        NodeLayer{bifur1_node, 4} = cp1;
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 2, bifur1_node);
        NodeLayer{bifur1_node, 5} = [dict_bif(idx_bif, 1), dict_pipe{idx_pipe, 1}];
        % record the element information around bifurcation
        bif_ele(index_bif, 3) = tmp_pointnumber;

        % * Save left terminal points and element (For extraction)

        [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
        AllPoint_extract = [AllPoint_extract; tmp_lpt(tmp_extract_p(:, 2), :)];
        PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];
        gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;
        lidx1_extract = [1:m1_extract] + m3_extract + m1_extract;
        lidx2_extract = [1:m1_extract];
        PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];

        for k = 1:m2_extract
            tmp_e_extract(k, 1:4) = extract_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
        end

        LayerElement_extract = [LayerElement_extract; tmp_e_extract(:, 1:4)];
        NodeLayer_extract{bifur1_node, 1} = tmp_lpt(tmp_extract_p(:, 2));
        NodeLayer_extract{bifur1_node, 2} = tmp_e_extract(:, 1:4);
        NodeLayer_extract{bifur1_node, 3} = tmp_e_extract(:, 1:4);
        NodeLayer_extract{bifur1_node, 4} = cp1;
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 2, bifur1_node);
        NodeLayer_extract{bifur1_node, 5} = [dict_bif(idx_bif, 1), dict_pipe{idx_pipe, 1}];
        PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * dict_bif(idx_bif, 1), ones(m1_extract, 1) * dict_pipe{idx_pipe, 1}]];

        % record the element information around bifurcation
        bif_ele_extract(index_bif, 3) = tmp_pointnumber_extract;

        % Save right terminal points and element
        if (termination(bifur2_node))
            tmp_label = ones(m1, 1);
            tmp_label = tip_label * tmp_label;
            tmp_label(boundary_point_index, :) = wall_label;
            tip_label = tip_label + 1;
        end

        for ii = 1:m1
            tmp_veloctity(ii, 1:3) = sv_bifur2 / norm(sv_bifur2) * norm(velocity_value(ii));
        end

        [tmp_pointnumber, tmp] = size(AllPoint);
        AllPoint = [AllPoint; tmp_rpt];
        AllLabel = [AllLabel; tmp_label];
        AllVelocity = [AllVelocity; tmp_veloctity];

        for k = 1:m2
            tmp_e(k, 1:4) = template_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
        end

        LayerElement = [LayerElement; tmp_e(:, 1:4)];
        NodeLayer{bifur2_node, 1} = tmp_rpt;
        NodeLayer{bifur2_node, 2} = tmp_e(:, 1:4);
        NodeLayer{bifur2_node, 3} = tmp_e(:, 1:4);
        NodeLayer{bifur2_node, 4} = cp1;
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 2, bifur2_node);
        NodeLayer{bifur2_node, 5} = [dict_bif(idx_bif, 1), dict_pipe{idx_pipe, 1}];
        % record the element information around bifurcation
        bif_ele(index_bif, 4) = tmp_pointnumber;

        % * Save right terminal points and element (For extraction)
        [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
        AllPoint_extract = [AllPoint_extract; tmp_rpt(tmp_extract_p(:, 2), :)];
        PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];
        gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;
        lidx1_extract = [1:m1_extract] + m3_extract + m1_extract * 2;
        lidx2_extract = [1:m1_extract];
        PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];

        for k = 1:m2_extract
            tmp_e_extract(k, 1:4) = extract_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
        end

        LayerElement_extract = [LayerElement_extract; tmp_e(:, 1:4)];
        NodeLayer_extract{bifur2_node, 1} = tmp_rpt(tmp_extract_p(:, 2));
        NodeLayer_extract{bifur2_node, 2} = tmp_e_extract(:, 1:4);
        NodeLayer_extract{bifur2_node, 3} = tmp_e_extract(:, 1:4);
        NodeLayer_extract{bifur2_node, 4} = cp1;
        [idx_bif, ~] = find(dict_bif(:, 2) == i);
        [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 2, bifur2_node);
        NodeLayer_extract{bifur2_node, 5} = [dict_bif(idx_bif, 1), dict_pipe{idx_pipe, 1}];
        PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * dict_bif(idx_bif, 1), ones(m1_extract, 1) * dict_pipe{idx_pipe, 1}]];

        % record the element information around bifurcation
        bif_ele_extract(index_bif, 4) = tmp_pointnumber_extract;

    end

    %% Label layers in pipe
    for index_sec = 1:n_sect
        [sect_ptnum, ~] = size(bif_term_pt{index_sec}');
        index_start_pt = bif_term_pt{index_sec}(1);
        index_end_pt = bif_term_pt{index_sec}(sect_ptnum);

        if index_start_pt == 1
            NodeLayer{index_start_pt, 5} = [1, 1];
            NodeLayer_extract{index_start_pt, 5} = [1, 1];
        end

        if termination(index_end_pt)
            [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 3, index_end_pt);
            NodeLayer{index_end_pt, 5} = [dict_pipe{idx_pipe, 1}, dict_pipe{idx_pipe, 1}];
            NodeLayer_extract{index_end_pt, 5} = [dict_pipe{idx_pipe, 1}, dict_pipe{idx_pipe, 1}];
        end

        for index_sec_pt = 2:sect_ptnum - 1
            idx_node = bif_term_pt{index_sec}(index_sec_pt);
            [idx_pipe, ~] = SearchPipeIdx(dict_pipe, 4, idx_node);
            [num_pipe, ~] = size(idx_pipe);

            if num_pipe == 1
                NodeLayer{idx_node, 5} = [dict_pipe{idx_pipe, 1}, dict_pipe{idx_pipe, 1}];
                NodeLayer_extract{idx_node, 5} = [dict_pipe{idx_pipe, 1}, dict_pipe{idx_pipe, 1}];
            else
                NodeLayer{idx_node, 5} = [dict_pipe{idx_pipe(2), 1}, dict_pipe{idx_pipe(1), 1}];
                NodeLayer_extract{idx_node, 5} = [dict_pipe{idx_pipe(2), 1}, dict_pipe{idx_pipe(1), 1}];
            end

        end

    end

    %% Calculate layers for bif-term branches and bif-bif branches
    if n_bif ~= 0

        for index_sec = 1:n_sect
            [sect_ptnum, ~] = size(bif_term_pt{index_sec}');
            index_start_pt = bif_term_pt{index_sec}(1);
            index_end_pt = bif_term_pt{index_sec}(sect_ptnum);
            n_insert_layer = sect_ptnum - 2;
            % bif-term segments
            if (index_start_pt == 1)
                n_insert_layer = n_insert_layer + 1;
                sec_vec_end = Segment_Vector{index_end_pt, id(index_end_pt)};
                ref_vec_end = NodeLayer{index_end_pt, 4};
                w = cross(sec_vec_end, ref_vec_end); w = w / norm(w);
                ref_vec_next = ref_vec_end / norm(ref_vec_end);

                for index_sec_pt = sect_ptnum:-1:2

                    j = bif_term_pt{index_sec}(index_sec_pt - 1);
                    i = bif_term_pt{index_sec}(index_sec_pt);
                    sv = Segment_Vector{i, j};
                    ref_vec_next = cross(w, sv); ref_vec_next = ref_vec_next / norm(ref_vec_next);

                    for ii = 1:m1
                        tmp_p(ii, :) = template_p(ii, 1) * ref_vec_next + template_p(ii, 2) * w;
                    end

                    tmp_p = tmp_p * d(j) / 2.;
                    w = cross(sv, ref_vec_next); w = w / norm(w);
                    [tmp_pointnumber, ~] = size(AllPoint);

                    tmp_label = ones(m1, 1);

                    if index_sec_pt == 2
                        tmp_label = tip_label * tmp_label;
                        tip_label = tip_label + 1;
                    else
                        tmp_label = in_point_label * tmp_label;
                    end

                    tmp_label(boundary_point_index, :) = wall_label;

                    for inode = 1:m1
                        tmp_p(inode, :) = tmp_p(inode, :) + location(j, :) - location(1, :);
                    end

                    for ii = 1:m1
                        tmp_veloctity(ii, 1:3) = sv / norm(sv) * norm(velocity_value(ii));
                    end

                    AllVelocity = [AllVelocity; tmp_veloctity];
                    AllPoint = [AllPoint; tmp_p];
                    AllLabel = [AllLabel; tmp_label];

                    for k = 1:m2
                        tmp_e(k, 1:4) = template_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
                    end

                    LayerElement = [LayerElement; tmp_e(:, 1:4)];
                    NodeLayer{j, 1} = tmp_p;
                    NodeLayer{j, 2} = tmp_e(:, 1:4);
                    NodeLayer{j, 3} = tmp_e(:, 1:4);

                    % * Extraction
                    [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
                    AllPoint_extract = [AllPoint_extract; tmp_p(extract_p(:, 2), :)];
                    PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];

                    % [idx_pipe, col] = find(dict_pipe(:, 2:end) == j);
                    [idx_pipe, row] = SearchPipeIdx(dict_pipe, 4, j);
                    num_pipe = size(idx_pipe, 1);
                    gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;

                    if num_pipe == 1
                        lidx1_extract = [1:m1_extract] + m1_extract * (row - 1);
                        lidx2_extract = [1:m1_extract] + m1_extract * (row - 1);
                    else
                        lidx1_extract = [1:m1_extract] + m1_extract * (row(2) - 1);
                        lidx2_extract = [1:m1_extract] + m1_extract * (row(1) - 1);
                    end

                    PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];
                    PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * NodeLayer_extract{j, 5}(1), ones(m1_extract, 1) * NodeLayer_extract{j, 5}(2)]];

                    for k = 1:m2_extract
                        tmp_e_extract(k, 1:4) = extract_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
                    end

                    LayerElement_extract = [LayerElement; tmp_e(:, 1:4)];
                    NodeLayer_extract{j, 1} = tmp_p(extract_p(:, 2));
                    NodeLayer_extract{j, 2} = tmp_e_extract(:, 1:4);
                    NodeLayer_extract{j, 3} = tmp_e_extract(:, 1:4);

                end

                continue;
            end

            if (termination(index_end_pt))
                n_insert_layer = n_insert_layer + 1;
                sec_vec_start = Segment_Vector{index_start_pt, id(index_start_pt)};
                ref_vec_start = NodeLayer{index_start_pt, 4};
                w = cross(sec_vec_start, ref_vec_start); w = w / norm(w);
                ref_vec_next = ref_vec_start / norm(ref_vec_start);

                for index_sec_pt = 2:sect_ptnum
                    j = bif_term_pt{index_sec}(index_sec_pt - 1);
                    i = bif_term_pt{index_sec}(index_sec_pt);
                    sv = Segment_Vector{i, j};
                    ref_vec_next = cross(w, sv); ref_vec_next = ref_vec_next / norm(ref_vec_next);
                    w = cross(sv, ref_vec_next); w = w / norm(w);

                    for ii = 1:m1
                        tmp_p(ii, :) = template_p(ii, 1) * ref_vec_next + template_p(ii, 2) * w;
                    end

                    tmp_p = tmp_p * d(i) / 2.;

                    [tmp_pointnumber, ~] = size(AllPoint);

                    tmp_label = ones(m1, 1);

                    if index_sec_pt == sect_ptnum
                        tmp_label = tip_label * tmp_label;
                        tip_label = tip_label + 1;
                    else
                        tmp_label = in_point_label * tmp_label;
                    end

                    tmp_label(boundary_point_index, :) = wall_label;

                    for inode = 1:m1
                        tmp_p(inode, :) = tmp_p(inode, :) + location(i, :) - location(1, :);
                    end

                    for ii = 1:m1
                        tmp_veloctity(ii, 1:3) = sv / norm(sv) * norm(velocity_value(ii));
                    end

                    AllVelocity = [AllVelocity; tmp_veloctity];
                    AllPoint = [AllPoint; tmp_p];
                    AllLabel = [AllLabel; tmp_label];

                    for k = 1:m2
                        tmp_e(k, 1:4) = template_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
                    end

                    LayerElement = [LayerElement; tmp_e(:, 1:4)];
                    NodeLayer{i, 1} = tmp_p;
                    NodeLayer{i, 2} = tmp_e(:, 1:4);
                    NodeLayer{i, 3} = tmp_e(:, 1:4);

                    % * Extraction
                    [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
                    AllPoint_extract = [AllPoint_extract; tmp_p(extract_p(:, 2), :)];
                    PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];
                    PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * NodeLayer_extract{i, 5}(1), ones(m1_extract, 1) * NodeLayer_extract{i, 5}(2)]];

                    [idx_pipe, row] = SearchPipeIdx(dict_pipe, 4, i);
                    num_pipe = size(idx_pipe, 1);
                    gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;

                    if num_pipe == 1
                        lidx1_extract = [1:m1_extract] + m1_extract * (row - 1);
                        lidx2_extract = [1:m1_extract] + m1_extract * (row - 1);
                    else
                        lidx1_extract = [1:m1_extract] + m1_extract * (row(2) - 1);
                        lidx2_extract = [1:m1_extract] + m1_extract * (row(1) - 1);
                    end

                    PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];

                    for k = 1:m2_extract
                        tmp_e_extract(k, 1:4) = extract_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
                    end

                    LayerElement_extract = [LayerElement; tmp_e_extract(:, 1:4)];
                    NodeLayer_extract{i, 1} = tmp_p(extract_p(:, 2));
                    NodeLayer_extract{i, 2} = tmp_e_extract(:, 1:4);
                    NodeLayer_extract{i, 3} = tmp_e_extract(:, 1:4);
                end

                continue;
            end

            % bif-bif segments
            sec_vec_start = Segment_Vector{index_start_pt, id(index_start_pt)};

            %         sec_vec_end=Segment_Vector{index_end_pt,id(index_end_pt)};
            sec_vec_end = Segment_Vector{find(trees{1}.dA(:, index_end_pt)), index_end_pt};

            ref_vec_start = NodeLayer{index_start_pt, 4};
            ref_vec_end = NodeLayer{index_end_pt, 4};

            n_start = cross(ref_vec_start, cross(sec_vec_start, ref_vec_start));
            n_start = n_start / norm(n_start);

            n_end = cross(ref_vec_end, cross(sec_vec_end, ref_vec_end));
            n_end = n_end / norm(n_end);

            ref_vec_start2end = RotateSurface(ref_vec_start, n_start, n_end);
            rotate_axis = cross(ref_vec_start2end, ref_vec_end); rotate_axis = rotate_axis / norm(rotate_axis);
            angle_total = acos(dot(ref_vec_start2end, ref_vec_end) / norm(ref_vec_start2end) / norm(ref_vec_end));
            angle_total_extract = angle_total;

            %         rotate_axis=cross(ref_vec_start,ref_vec_end);
            %         angle_total=acos(dot(ref_vec_start,ref_vec_end)/norm(ref_vec_start)/norm(ref_vec_end));

            if angle_total > pi / 4 && angle_total <= 3 * pi / 4
                angle_total = angle_total - pi / 2;
                %             if dot(rotate_axis,sec_vec_end)>0
                if dot(rotate_axis, n_end) > 0
                    %                     if dot(rotate_axis,[0 0 1])>0
                    for index_ele = 1:m2 / 4
                        tmp_end_element(index_ele + m2 / 4 * 0, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 3, [3 4 1 2]);
                        tmp_end_element(index_ele + m2 / 4 * 1, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 0, :);
                        tmp_end_element(index_ele + m2 / 4 * 2, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 1, [3 4 1 2]);
                        tmp_end_element(index_ele + m2 / 4 * 3, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 2, :);
                    end

                    NodeLayer{index_end_pt, 3} = tmp_end_element;
                    %             elseif dot(rotate_axis,sec_vec_end)<0
                elseif dot(rotate_axis, n_end) < 0
                    %                     elseif dot(rotate_axis,[0 0 1])<0
                    for index_ele = 1:m2 / 4
                        tmp_end_element(index_ele + m2 / 4 * 0, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 1, :);
                        tmp_end_element(index_ele + m2 / 4 * 1, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 2, [3 4 1 2]);
                        tmp_end_element(index_ele + m2 / 4 * 2, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 3, :);
                        tmp_end_element(index_ele + m2 / 4 * 3, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 0, [3 4 1 2]);
                    end

                    NodeLayer{index_end_pt, 3} = tmp_end_element;
                end

            elseif angle_total > 3 * pi / 4 && angle_total <= pi
                angle_total = angle_total - pi;

                for index_ele = 1:m2 / 4
                    tmp_end_element(index_ele + m2 / 4 * 0, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 2, [3 4 1 2]);
                    tmp_end_element(index_ele + m2 / 4 * 1, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 3, [3 4 1 2]);
                    tmp_end_element(index_ele + m2 / 4 * 2, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 0, [3 4 1 2]);
                    tmp_end_element(index_ele + m2 / 4 * 3, :) = NodeLayer{index_end_pt, 2}(index_ele + m2 / 4 * 1, [3 4 1 2]);
                end

                NodeLayer{index_end_pt, 3} = tmp_end_element;
            end

            % *Extraction
            angle_total = angle_total_extract;

            if angle_total > pi / 4 && angle_total <= 3 * pi / 4
                angle_total = angle_total - pi / 2;
                %             if dot(rotate_axis,sec_vec_end)>0
                if dot(rotate_axis, n_end) > 0
                    %                     if dot(rotate_axis,[0 0 1])>0
                    for index_ele = 1:m2_extract / 4
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 0, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 3, :);
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 1, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 0, :);
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 2, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 1, :);
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 3, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 2, :);
                    end

                    NodeLayer_extract{index_end_pt, 3} = tmp_end_element_extract;
                    %             elseif dot(rotate_axis,sec_vec_end)<0
                elseif dot(rotate_axis, n_end) < 0
                    %                     elseif dot(rotate_axis,[0 0 1])<0
                    for index_ele = 1:m2_extract / 4
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 0, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 1, :);
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 1, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 2, :);
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 2, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 3, :);
                        tmp_end_element_extract(index_ele + m2_extract / 4 * 3, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 0, :);
                    end

                    NodeLayer_extract{index_end_pt, 3} = tmp_end_element_extract;
                end

            elseif angle_total > 3 * pi / 4 && angle_total <= pi
                angle_total = angle_total - pi;

                for index_ele = 1:m2_extract / 4
                    tmp_end_element_extract(index_ele + m2_extract / 4 * 0, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 2, :);
                    tmp_end_element_extract(index_ele + m2_extract / 4 * 1, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 3, :);
                    tmp_end_element_extract(index_ele + m2_extract / 4 * 2, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 0, :);
                    tmp_end_element_extract(index_ele + m2_extract / 4 * 3, :) = NodeLayer_extract{index_end_pt, 2}(index_ele + m2_extract / 4 * 1, :);
                end

                NodeLayer_extract{index_end_pt, 3} = tmp_end_element_extract;
            end

            angle_per = angle_total / (n_insert_layer + 1);

            fprintf('sec_index=%d angle_total=%f angle_per=%f dot(rotate_axis,n_end)=%f\n', index_sec, angle_total * 180 / pi, angle_per * 180 / pi, dot(rotate_axis, n_end));

            ref_temp2start = RotateSurface([1 0 0], [0 0 1], n_start);
            rotate_axis_start = cross(ref_temp2start, ref_vec_start);
            angle_start = acos(dot(ref_temp2start, ref_vec_start) / norm(ref_vec_start) / norm(ref_temp2start));

            for index_sec_pt = 2:sect_ptnum - 1
                j = bif_term_pt{index_sec}(index_sec_pt - 1);
                i = bif_term_pt{index_sec}(index_sec_pt);
                sv = Segment_Vector{i, j};

                tmp_p = template_p * d(i) / 2.;

                tmp_p = RotateSurface(tmp_p, [0 0 1], n_start);
                tmp_p = RotateAroundAxis(tmp_p, rotate_axis_start, angle_start);

                if dot(rotate_axis, n_end) > 0
                    tmp_p = RotateAroundAxis(tmp_p, n_start, angle_per * (index_sec_pt - 1));
                elseif dot(rotate_axis, n_end) < 0
                    tmp_p = RotateAroundAxis(tmp_p, -n_start, angle_per * (index_sec_pt - 1));
                end

                tmp_p = RotateSurface(tmp_p, n_start, sv);

                [tmp_pointnumber, ~] = size(AllPoint);

                tmp_label = ones(m1, 1);
                tmp_label = in_point_label * tmp_label;
                tmp_label(boundary_point_index, :) = wall_label;

                for inode = 1:m1
                    tmp_p(inode, :) = tmp_p(inode, :) + location(i, :) - location(1, :);
                end

                for ii = 1:m1
                    tmp_veloctity(ii, 1:3) = sv / norm(sv) * norm(velocity_value(ii));
                end

                AllVelocity = [AllVelocity; tmp_veloctity];
                AllPoint = [AllPoint; tmp_p];
                AllLabel = [AllLabel; tmp_label];

                for k = 1:m2
                    tmp_e(k, 1:4) = template_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
                end

                LayerElement = [LayerElement; tmp_e(:, 1:4)];
                NodeLayer{i, 1} = tmp_p;
                NodeLayer{i, 2} = tmp_e(:, 1:4);
                NodeLayer{i, 3} = tmp_e(:, 1:4);

                % * Extraction
                [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
                AllPoint_extract = [AllPoint_extract; tmp_p(extract_p(:, 2), :)];
                PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];
                [idx_pipe, row] = SearchPipeIdx(dict_pipe, 4, i);
                num_pipe = size(idx_pipe, 1);
                gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;

                if num_pipe == 1
                    lidx1_extract = [1:m1_extract] + m1_extract * (row - 1);
                    lidx2_extract = [1:m1_extract] + m1_extract * (row - 1);
                else
                    lidx1_extract = [1:m1_extract] + m1_extract * (row(2) - 1);
                    lidx2_extract = [1:m1_extract] + m1_extract * (row(1) - 1);
                end

                PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];
                PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * NodeLayer_extract{i, 5}(1), ones(m1_extract, 1) * NodeLayer_extract{i, 5}(2)]];

                for k = 1:m2_extract
                    tmp_e_extract(k, 1:4) = extract_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
                end

                LayerElement_extract = [LayerElement_extract; tmp_e_extract(:, 1:4)];
                NodeLayer_extract{i, 1} = tmp_p(extract_p(:, 2));
                NodeLayer_extract{i, 2} = tmp_e_extract(:, 1:4);
                NodeLayer_extract{i, 3} = tmp_e_extract(:, 1:4);
            end

        end

    else
        %deal with pipe geometry (without any bifurcation)
        [sect_ptnum, ~] = size(bif_term_pt{1}');
        index_start_pt = bif_term_pt{1}(1);
        index_end_pt = bif_term_pt{1}(sect_ptnum);
        n_insert_layer = sect_ptnum;

        for index_sec_pt = 1:sect_ptnum

            if index_sec_pt == 1
                j = bif_term_pt{index_sec}(1);
                i = bif_term_pt{index_sec}(2);
                sv = Segment_Vector{i, j};
                ref_vec_start = RotateSurface([1, 0, 0], [0 0 1], sv);
                ref_vec_next = ref_vec_start / norm(ref_vec_start);
                w = cross(sv, ref_vec_start); w = w / norm(w);
                i = 1;
            else
                j = bif_term_pt{index_sec}(index_sec_pt - 1);
                i = bif_term_pt{index_sec}(index_sec_pt);
                sv = Segment_Vector{i, j};
            end

            ref_vec_next = cross(w, sv); ref_vec_next = ref_vec_next / norm(ref_vec_next);
            w = cross(sv, ref_vec_next); w = w / norm(w);

            tmp_p = template_p * d(i) / 2.;

            if index_sec_pt == 1
                tmp_p = RotateSurface(tmp_p, [0 0 1], sv);
            else

                for ii = 1:m1
                    tmp_p(ii, :) = tmp_p(ii, 1) * ref_vec_next + tmp_p(ii, 2) * w;
                end

            end

            [tmp_pointnumber, tmp] = size(AllPoint);

            tmp_label = ones(m1, 1);

            if index_sec_pt == sect_ptnum || index_sec_pt == 1
                tmp_label = tip_label * tmp_label;
                tip_label = tip_label + 1;
            else
                tmp_label = in_point_label * tmp_label;
            end

            tmp_label(boundary_point_index, :) = wall_label;

            for inode = 1:m1
                tmp_p(inode, :) = tmp_p(inode, :) + location(i, :) - location(1, :);
            end

            for ii = 1:m1
                tmp_veloctity(ii, 1:3) = sv / norm(sv) * norm(velocity_value(ii));
            end

            AllVelocity = [AllVelocity; tmp_veloctity];
            AllPoint = [AllPoint; tmp_p];
            AllLabel = [AllLabel; tmp_label];

            for k = 1:m2
                tmp_e(k, 1:4) = template_e(k, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
            end

            LayerElement = [LayerElement; tmp_e(:, 1:4)];
            NodeLayer{i, 1} = tmp_p;
            NodeLayer{i, 2} = tmp_e(:, 1:4);
            NodeLayer{i, 3} = tmp_e(:, 1:4);

            % * Extraction
            [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
            AllPoint_extract = [AllPoint; tmp_p(extract_p(:, 2), :)];
            PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];
            [idx_pipe, row] = SearchPipeIdx(dict_pipe, 4, i);
            num_pipe = size(idx_pipe, 1);
            gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;

            if num_pipe == 1
                lidx1_extract = [1:m1_extract] + m1_extract * (row - 1);
                lidx2_extract = [1:m1_extract] + m1_extract * (row - 1);
            else
                lidx1_extract = [1:m1_extract] + m1_extract * (row(2) - 1);
                lidx2_extract = [1:m1_extract] + m1_extract * (row(1) - 1);
            end

            PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];
            PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * NodeLayer_extract{i, 5}(1), ones(m1_extract, 1) * NodeLayer_extract{i, 5}(2)]];

            for k = 1:m2_extract
                tmp_e_extract(k, 1:4) = extract_e(k, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
            end

            LayerElement_extract = [LayerElement; tmp_e(:, 1:4)];
            NodeLayer_extract{i, 1} = tmp_p(extract_p(:, 2));
            NodeLayer_extract{i, 2} = tmp_e_extract(:, 1:4);
            NodeLayer_extract{i, 3} = tmp_e_extract(:, 1:4);
        end

    end

    %% Calculate the points and index of the layers in each segment
    for index_sec = 1:n_sect
        [sect_ptnum, tmp] = size(sect_pt{index_sec}');

        for index_sec_pt = 1:sect_ptnum - 1
            j = sect_pt{index_sec}(index_sec_pt);
            i = sect_pt{index_sec}(index_sec_pt + 1);
            segment_layer = n_layer(i, j);
            sv = Segment_Vector{i, j};

            if (branch(j))
                sv = Segment_Vector{i, j};

                if (ijk_label(i) == 1)
                    tmp_start_e = NodeLayer{j, 3};
                    tmp_start_p = BranchLayerPoint{j, 2};
                elseif (ijk_label(i) == 2)
                    tmp_start_e = NodeLayer{j, 4};
                    tmp_start_p = BranchLayerPoint{j, 3};
                end

            else
                tmp_start_p = NodeLayer{j, 1};
                tmp_start_e = NodeLayer{j, 2};
            end

            if (branch(i))
                tmp_end_p = BranchLayerPoint{i, 1};
                tmp_end_e = NodeLayer{i, 2};
            else
                tmp_end_p = NodeLayer{i, 1};
                tmp_end_e = NodeLayer{i, 3};
            end

            % * Extraction
            if (branch(j))
                sv = Segment_Vector{i, j};

                if (ijk_label(i) == 1)
                    tmp_start_e_extract = NodeLayer_extract{j, 3};
                    tmp_start_p_extract = BranchLayerPoint_extract{j, 2};
                elseif (ijk_label(i) == 2)
                    tmp_start_e_extract = NodeLayer_extract{j, 4};
                    tmp_start_p_extract = BranchLayerPoint_extract{j, 3};
                end

            else
                tmp_start_p_extract = NodeLayer_extract{j, 1};
                tmp_start_e_extract = NodeLayer_extract{j, 2};
            end

            if (branch(i))
                tmp_end_p_extract = BranchLayerPoint_extract{i, 1};
                tmp_end_e_extract = NodeLayer_extract{i, 2};
            else
                tmp_end_p_extract = NodeLayer_extract{i, 1};
                tmp_end_e_extract = NodeLayer_extract{i, 3};
            end

            % * Determine element label
            ele_label_diff = NodeLayer_extract{i, 5} - NodeLayer_extract{j, 5};
            % [row, ~]= find(ele_label_diff == 0, 1);
            row = find(ele_label_diff == 0, 1);

            if isempty(row)
                fprintf("Found empty row: idx_sec: %d; idx_sec_pt: %d;", index_sec, index_sec_pt)
            end

            if segment_layer == 1
                [tmp_pointnumber, ~] = size(AllPoint);
                AllElement = [AllElement; tmp_start_e, tmp_end_e];

                % * Extraction
                [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
                AllElement_extract = [AllElement_extract; tmp_start_e_extract, tmp_end_e_extract];

                AllElementLabel = [AllElementLabel; ones(size(tmp_start_e, 1), 1) * NodeLayer_extract{i, 5}(row)];
                AllElementLabel_extract = [AllElementLabel_extract; ones(size(tmp_start_e_extract, 1), 1) * NodeLayer_extract{i, 5}(row)];

            else

                for k = 1:segment_layer - 1
                    [tmp_pointnumber, ~] = size(AllPoint);
                    [element_index, ~] = size(AllElement);
                    [layer_index, ~] = size(LayerElement);

                    % record the element information around bifurcation
                    if (branch(i) && k == segment_layer - 1)
                        index_bif = find(bif_pt(:, 1) == i);
                        bif_ele(index_bif, 2) = tmp_pointnumber;
                    end

                    if (branch(j) && k == 1)
                        index_bif = find(bif_pt(:, 1) == j);

                        if (ijk_label(i) == 1)
                            bif_ele(index_bif, 3) = tmp_pointnumber;
                        elseif (ijk_label(i) == 2)
                            bif_ele(index_bif, 4) = tmp_pointnumber;
                        end

                    end

                    for inode = 1:m1
                        tmp_p(inode, :) = (tmp_end_p(inode, :) * k + tmp_start_p(inode, :) * (segment_layer - k)) / segment_layer;
                    end

                    tmp_label = ones(m1, 1);
                    tmp_label = in_point_label * tmp_label;
                    tmp_label(boundary_point_index, :) = wall_label;

                    for ii = 1:m1
                        tmp_veloctity(ii, 1:3) = sv / norm(sv) * norm(velocity_value(ii));
                    end

                    AllVelocity = [AllVelocity; tmp_veloctity];

                    AllPoint = [AllPoint; tmp_p];
                    AllLabel = [AllLabel; tmp_label];

                    for iele = 1:m2
                        tmp_e(iele, 1:4) = template_e(iele, 2:5) + tmp_pointnumber * [1, 1, 1, 1];
                    end

                    LayerElement = [LayerElement; tmp_e(:, 1:4)];

                    if (k == 1)
                        AllElement = [AllElement; tmp_start_e, tmp_e(:, 1:4)];
                        AllElement = [AllElement; tmp_e(:, 1:4), zeros(m2, 4)];
                        AllElementLabel = [AllElementLabel; ones(size(tmp_start_e, 1), 1) * NodeLayer_extract{i, 5}(row)];
                        AllElementLabel = [AllElementLabel; ones(size(tmp_e, 1), 1) * NodeLayer_extract{i, 5}(row)];
                    else
                        AllElement((element_index + 1 - m2):element_index, 5:8) = tmp_e(:, 1:4);
                        AllElement = [AllElement; tmp_e(:, 1:4), zeros(m2, 4)];
                        AllElementLabel = [AllElementLabel; ones(size(tmp_e, 1), 1) * NodeLayer_extract{i, 5}(row)];
                    end

                    [element_index, ~] = size(AllElement);
                    AllElement((element_index + 1 - m2):element_index, 5:8) = tmp_end_e;

                    % * Extraction
                    [tmp_pointnumber_extract, ~] = size(AllPoint_extract);
                    [element_index_extract, ~] = size(AllElement_extract);
                    [layer_index_extract, ~] = size(LayerElement_extract);

                    % record the element information around bifurcation
                    if (branch(i) && k == segment_layer - 1)
                        index_bif = find(bif_pt(:, 1) == i);
                        bif_ele_extract(index_bif, 2) = tmp_pointnumber_extract;
                    end

                    if (branch(j) && k == 1)
                        index_bif = find(bif_pt(:, 1) == j);

                        if (ijk_label(i) == 1)
                            bif_ele_extract(index_bif, 3) = tmp_pointnumber_extract;
                        elseif (ijk_label(i) == 2)
                            bif_ele_extract(index_bif, 4) = tmp_pointnumber_extract;
                        end

                    end

                    % for inode=1:m1
                    %     tmp_p(inode,:)=(tmp_end_p(inode,:)*k+tmp_start_p(inode,:)*(segment_layer-k))/segment_layer;
                    % end

                    % tmp_label=ones(m1,1);
                    % tmp_label=in_point_label*tmp_label;
                    % tmp_label(boundary_point_index,:)=wall_label;

                    AllPoint_extract = [AllPoint_extract; tmp_p(extract_p(:, 2), :)];
                    PointMapping_extract = [PointMapping_extract; extract_p(:, 2) + tmp_pointnumber];
                    gidx_extract = [1:m1_extract] + tmp_pointnumber_extract;

                    if (branch(i))
                        lidx1_extract = [1:m1_extract] + m3_extract + m1_extract * (3 + k - 1);
                        lidx2_extract = [1:m1_extract] + m3_extract + m1_extract * (3 + k - 1);
                    end

                    if (branch(j))
                        index_bif = find(bif_pt(:, 1) == j);
                        par_node = bif_pt(index_bif, 2);
                        bifur1_node = bif_pt(index_bif, 3);
                        bifur2_node = bif_pt(index_bif, 4);

                        if (i == bifur1_node)
                            lidx1_extract = [1:m1_extract] + m3_extract + m1_extract * (3 + k - 1 + n_layer(j, par_node) - 1);
                            lidx2_extract = [1:m1_extract] + m3_extract + m1_extract * (3 + k - 1 + n_layer(j, par_node) - 1);
                        elseif (i == bifur2_node)
                            lidx1_extract = [1:m1_extract] + m3_extract + m1_extract * (3 + k - 1 + n_layer(j, par_node) - 1 + n_layer(bifur1_node, j) - 1);
                            lidx2_extract = [1:m1_extract] + m3_extract + m1_extract * (3 + k - 1 + n_layer(j, par_node) - 1 + n_layer(bifur1_node, j) - 1);
                        end

                    end

                    PointMapping_simulator = [PointMapping_simulator; [gidx_extract', lidx1_extract', lidx2_extract']];
                    % PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * NodeLayer_extract{j, 5}(2), ones(m1_extract, 1) * NodeLayer_extract{i, 5}(1)]];
                    PointLabel_simulator = [PointLabel_simulator; [ones(m1_extract, 1) * NodeLayer_extract{i, 5}(row), ones(m1_extract, 1) * NodeLayer_extract{i, 5}(row)]];

                    % AllLabel=[AllLabel;tmp_label];
                    for iele = 1:m2_extract
                        tmp_e_extract(iele, 1:4) = extract_e(iele, 2:5) + tmp_pointnumber_extract * [1, 1, 1, 1];
                    end

                    LayerElement_extract = [LayerElement; tmp_e_extract(:, 1:4)];

                    if (k == 1)
                        AllElement_extract = [AllElement_extract; tmp_start_e_extract, tmp_e_extract(:, 1:4)];
                        AllElement_extract = [AllElement_extract; tmp_e_extract(:, 1:4), zeros(m2_extract, 4)];
                        AllElementLabel_extract = [AllElementLabel_extract; ones(size(tmp_start_e_extract, 1), 1) * NodeLayer_extract{i, 5}(row)];
                        AllElementLabel_extract = [AllElementLabel_extract; ones(size(tmp_e_extract, 1), 1) * NodeLayer_extract{i, 5}(row)];

                    else
                        AllElement_extract((element_index_extract + 1 - m2_extract):element_index_extract, 5:8) = tmp_e_extract(:, 1:4);
                        AllElement_extract = [AllElement_extract; tmp_e_extract(:, 1:4), zeros(m2_extract, 4)];
                        AllElementLabel_extract = [AllElementLabel_extract; ones(size(tmp_e_extract, 1), 1) * NodeLayer_extract{i, 5}(row)];
                    end

                    [element_index_extract, ~] = size(AllElement_extract);
                    AllElement_extract((element_index_extract + 1 - m2_extract):element_index_extract, 5:8) = tmp_end_e_extract;

                end

            end

        end

        % Deal with bifurcation point
        for index_bif = 1:n_bif
            startpt_b = bif_ele(index_bif, 1);
            startpt_i = bif_ele(index_bif, 2);
            startpt_j = bif_ele(index_bif, 3);
            startpt_k = bif_ele(index_bif, 4);

            [num_bc, tmp] = size(leftbc_index);

            for i = 1:num_bc
                AllPoint(startpt_b + botbc_index(i), :) = PointAlign(AllPoint(startpt_j + rightbc_index(i), :), AllPoint(startpt_k + leftbc_index(i), :), AllPoint(startpt_b + botbc_index(i), :));
                AllPoint(startpt_b + leftbc_index(i), :) = PointAlign(AllPoint(startpt_i + leftbc_index(i), :), AllPoint(startpt_j + leftbc_index(i), :), AllPoint(startpt_b + leftbc_index(i), :));
                AllPoint(startpt_b + rightbc_index(i), :) = PointAlign(AllPoint(startpt_i + rightbc_index(i), :), AllPoint(startpt_k + rightbc_index(i), :), AllPoint(startpt_b + rightbc_index(i), :));
            end

            for i = 1:2
                AllPoint(startpt_b + extra_pt(i, 1), :) = Projection(AllPoint(startpt_b + extra_pt(i, 2), :), AllPoint(startpt_b + extra_pt(i, 3), :), AllPoint(startpt_b + extra_pt(i, 4), :), AllPoint(startpt_b + extra_pt(i, 1), :));
                AllPoint(startpt_i + extra_pt(i, 1), :) = Projection(AllPoint(startpt_b + extra_pt(i, 2), :), AllPoint(startpt_b + extra_pt(i, 3), :), AllPoint(startpt_b + extra_pt(i, 4), :), AllPoint(startpt_i + extra_pt(i, 1), :));
                AllPoint(startpt_j + extra_pt(i, 1), :) = Projection(AllPoint(startpt_b + extra_pt(i, 2), :), AllPoint(startpt_b + extra_pt(i, 3), :), AllPoint(startpt_b + extra_pt(i, 4), :), AllPoint(startpt_j + extra_pt(i, 1), :));
                AllPoint(startpt_k + extra_pt(i, 1), :) = Projection(AllPoint(startpt_b + extra_pt(i, 2), :), AllPoint(startpt_b + extra_pt(i, 3), :), AllPoint(startpt_b + extra_pt(i, 4), :), AllPoint(startpt_k + extra_pt(i, 1), :));
            end

        end

    end

    %% * Extraction
    dlmwrite(mapping_simulator_output, PointMapping_simulator - 1, 'Delimiter', '\t');

    % Point
    AllPoint_extract = AllPoint(PointMapping_extract, :);

    % Pipe simulator
    n_pipe_sim = size(dict_pipe, 1);

    for idx_sim = 1:n_pipe_sim
        idx_global_sim = dict_pipe{idx_sim, 1};
        tmp_pt_idx1 = PointLabel_simulator(:, 1) == idx_global_sim;
        tmp_pt_idx2 = PointLabel_simulator(:, 2) == idx_global_sim;
        tmp_pt_idx_and = tmp_pt_idx1 & tmp_pt_idx2;

        C2Smapping = [PointMapping_simulator(tmp_pt_idx1 == 1, [1, 2]); PointMapping_simulator((tmp_pt_idx2 - tmp_pt_idx_and) == 1, [1, 3])];
        S2Dmapping(C2Smapping(:, 2)) = PointMapping_extract(C2Smapping(:, 1));

        pt_sim(C2Smapping(:, 2), :) = AllPoint_extract(C2Smapping(:, 1), :);

        C2Smapping(:, 2) = C2Smapping(:, 2) - 1;
        tmp_ele_idx = (AllElementLabel_extract == idx_global_sim);
        ele_sim_global = AllElement_extract(tmp_ele_idx == 1, :);

        idx_sim
        LUT(C2Smapping(:, 1)) = C2Smapping(:, 2);
        ele_sim_local = LUT(ele_sim_global + 1);

        % Output the mesh of the extracted simulator
        fname_sim = [sim_path, 'simulator_', num2str(idx_global_sim), '.vtk'];
        WriteVTK(fname_sim, pt_sim, ele_sim_local, [], []);

        % Output the edge of the extracted simulator in vtk format
        pt_sample = [pt_sim(:, 1) - pt_sim(1, 1), pt_sim(:, 2) - pt_sim(1, 2), pt_sim(:, 3) - pt_sim(1, 3)];
        edge = MeshToEdge(ele_sim_local);
        fname_sim_edge_vtk = [sim_path, 'simulator_', num2str(idx_global_sim), '_edge.vtk'];
        WriteSimulatorSample(fname_sim_edge_vtk, pt_sample, edge);

        % Output the edge of the extracted simulator in txt format
        fname_sim_edge_txt = [sim_path, 'simulator_', num2str(idx_global_sim), '_edge.txt'];
        dlmwrite(fname_sim_edge_txt, edge, 'Delimiter', '\t', 'precision', '%d');

        % Output the mapping (Extracted simple mesh to the simulator)
        fname_sim_C2Smapping = [sim_path, 'simulator_', num2str(idx_global_sim), '_C2Smapping.txt'];
        C2Smapping(:, 1) = C2Smapping(:, 1) - 1;
        dlmwrite(fname_sim_C2Smapping, C2Smapping, 'Delimiter', '\t', 'precision', '%d');

        % Output the mapping (Extracted simple mesh to the simulator)
        fname_sim_S2Dmapping = [sim_path, 'simulator_', num2str(idx_global_sim), '_S2Dmapping.txt'];
        dlmwrite(fname_sim_S2Dmapping, S2Dmapping - 1, 'Delimiter', ',', 'precision', '%d');
    end

    % Bif simulator
    n_bif_sim = size(dict_bif, 1);

    for idx_sim = 1:n_bif_sim
        idx_global_sim = dict_bif(idx_sim, 1);
        tmp_pt_idx1 = PointLabel_simulator(:, 1) == idx_global_sim;
        tmp_pt_idx2 = PointLabel_simulator(:, 2) == idx_global_sim;
        tmp_pt_idx_and = tmp_pt_idx1 & tmp_pt_idx2;

        C2Smapping = [PointMapping_simulator(tmp_pt_idx1 == 1, [1, 2]); PointMapping_simulator((tmp_pt_idx2 - tmp_pt_idx_and) == 1, [1, 3])];
        S2Dmapping(C2Smapping(:, 2)) = PointMapping_extract(C2Smapping(:, 1));

        pt_bif(C2Smapping(:, 2), :) = AllPoint_extract(C2Smapping(:, 1), :);

        C2Smapping(:, 2) = C2Smapping(:, 2) - 1;
        tmp_ele_idx = (AllElementLabel_extract == idx_global_sim);
        ele_sim_global = AllElement_extract(tmp_ele_idx == 1, :);

        LUT(C2Smapping(:, 1)) = C2Smapping(:, 2);
        ele_sim_local = LUT(ele_sim_global + 1);

        % Output the mesh of the extracted simulator
        fname_sim = [sim_path, 'simulator_', num2str(idx_global_sim), '.vtk'];
        WriteVTK(fname_sim, pt_bif, ele_sim_local, [], []);

        % Output the edge of the extracted simulator in vtk format
        pt_sample = [pt_bif(:, 1) - pt_bif(1, 1), pt_bif(:, 2) - pt_bif(1, 2), pt_bif(:, 3) - pt_bif(1, 3)];
        edge = MeshToEdge(ele_sim_local);
        fname_sim_edge_vtk = [sim_path, 'simulator_', num2str(idx_global_sim), '_edge.vtk'];
        WriteSimulatorSample(fname_sim_edge_vtk, pt_sample, edge);

        % Output the edge of the extracted simulator in txt format
        fname_sim_edge_txt = [sim_path, 'simulator_', num2str(idx_global_sim), '_edge.txt'];
        dlmwrite(fname_sim_edge_txt, edge, 'Delimiter', '\t', 'precision', '%d');

        % Output the mapping (Extracted simple mesh to the simulator)
        fname_sim_mapping = [sim_path, 'simulator_', num2str(idx_global_sim), '_C2Smapping.txt'];
        C2Smapping(:, 1) = C2Smapping(:, 1) - 1;
        dlmwrite(fname_sim_mapping, C2Smapping, 'Delimiter', '\t', 'precision', '%d');

        % Output the mapping (Extracted simple mesh to the simulator)
        fname_sim_S2Dmapping = [sim_path, 'simulator_', num2str(idx_global_sim), '_S2Dmapping.txt'];
        dlmwrite(fname_sim_S2Dmapping, S2Dmapping - 1, 'Delimiter', ',', 'precision', '%d');

    end

    %% Output Mesh
    %write the vtk file for view
    [n_point, ~] = size(AllPoint);
    [n_element, ~] = size(AllElement);
    [n_element_layer, ~] = size(LayerElement);
    fid1 = fopen(velocity_output, 'w');

    for ii = 1:n_point
        fprintf(fid1, '%f %f %f\n', AllVelocity(ii, 1:3));
    end

    fclose(fid1);

    %write the mesh vtk file
    WriteVTK(hex_output, AllPoint, AllElement, AllLabel, AllElementLabel);
    %     WriteVTK(hex_output, AllPoint, AllElement, AllLabel, []);

    %write the mesh info file (num of nodes and elements)
    fid1 = fopen(hex_info_output, 'w');
    fprintf(fid1, '%d %d\n', [n_point, n_element]);
    fclose(fid1);

    % *write the extracted mesh vtk file
    WriteVTK(hex_extract_output, AllPoint_extract, AllElement_extract, [], AllElementLabel_extract);
    %     WriteVTK(hex_extract_output, AllPoint_extract, AllElement_extract, [], []);

    % *write extracted point mapping
    dlmwrite(mapping_extract_output, PointMapping_extract' - 1, 'Delimiter', ',', 'precision', '%d');
    
    % *write extracted edge vtk file
    AllEdge_extract = MeshToEdge(AllElement_extract);
    hex_extract_edgevtk_output = [io_path, '//edge_extract.vtk'];
    WriteSimulatorSample(hex_extract_edgevtk_output, AllPoint_extract, AllEdge_extract);

    % *write extracted edge txt file
    hex_extract_edgetxt_output = [io_path, '//edge_extract.txt'];
    dlmwrite(hex_extract_edgetxt_output, AllEdge_extract, 'Delimiter', '\t', 'precision', '%d');


end
