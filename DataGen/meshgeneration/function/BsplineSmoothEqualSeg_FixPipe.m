function [XYZ_smooth, D_smooth, tangent_vec, n_chop] = BsplineSmoothEqualSeg_FixPipe(XYZ, D, mode, n_sample, n_seg_per_chop, len_chop)
    %Use the skeleton nodes as control points to construct B-spline
    % and connect the points on the spline to create new series of segments.
    % Input
    % -----
    % - XYZ: original skeleton nodes coordinates
    % - D:   original skeleton nodes diameters
    % - n_sample: the number of sample points used to compute the length of spline
    % - n_seg_per_chop: the number of segments in each short branch
    % - len_chop: the largest length of the short branch
    % - mode: -- 1 for branch with both bifurcation ends
    %         -- 2 for branch with one end as termination
    %         -- 3 for the branch with start node
    %
    % Outputs
    % -------
    % - XYZ_smooth: smoothed skeleton nodes coordinates
    % - D_smooth: smoothed skeleton nodes diameters
    % - tangent_vec: the tangent vector at each node
    % - n_chop: the number of short branches

    [n_cpt, ~] = size(XYZ);
    d = 0;

    for i = 2:n_cpt
        d = d + norm(XYZ(i, :) - XYZ(i - 1, :));
    end

    u = zeros(n_cpt, 1);

    for k = 2:n_cpt - 1
        u(k) = u(k - 1) + norm(XYZ(k, :) - XYZ(k - 1, :)) / d;
    end

    u(n_cpt) = 1;

    if n_cpt > 3
        uu = u';
        knot = [0 0 0 0 uu(3:end - 2) 1 1 1 1];
    elseif n_cpt == 3
        knot = [0, 0, 0, 1, 1, 1];
    else
        knot = [0, 0, 1, 1];
    end

    % Compute sampled point on spline
    sp_xyz = spmak(knot, XYZ');

    % Set sample points to compute Lookup Table for parametric coordinate t vs length
    if nargin < 4
        n_sample = 201; % Can be set as input parameter
        n_seg_per_chop = 7; % Can be set as input parameter
        len_chop = 14;
    end

    alpha = 1.5; % control distance from branch point to the center point at the first branch plane
    n_chop = 1;

    sample_t = linspace(0, 1, n_sample);
    dt = 1.0 / (n_sample - 1);

    % Compute Lookup Table for t vs length
    l_spline = [0];
    XYZ_sample = fnval(sp_xyz, sample_t);
    seg_vec = XYZ_sample(:, 2:n_sample) - XYZ_sample(:, 1:n_sample - 1);
    seg_length = zeros(n_sample - 1, 1);

    for i = 1:n_sample - 1
        seg_length(i) = norm(seg_vec(1:3, i));
        l_spline = [l_spline, l_spline(end) + seg_length(i)];
    end

    if (l_spline(end) > len_chop)
        n_chop = round(l_spline(end) / len_chop);
    end

    n_seg = n_seg_per_chop * n_chop;

    l_spline(end)
    seg_desire = linspace(0, l_spline(end), n_seg + 1);

    if (mode == 1)% branch -> branch
        seg_desire = linspace(D(1) * alpha, (l_spline(end) - D(end)* alpha), n_seg + 1);
        seg_desire = [0, seg_desire, l_spline(end)];
    elseif mode == 2% branch -> terminal
        seg_desire = linspace(D(1)* alpha, l_spline(end), n_seg + 1);
        seg_desire = [0, seg_desire];
    elseif mode == 3% root -> branch
        seg_desire = linspace(0, (l_spline(end) - D(end)* alpha), n_seg + 1);
        seg_desire = [seg_desire, l_spline(end)];
    elseif mode == 4% single pipe
        seg_desire = linspace(0, l_spline(end), n_seg + 1);
    end

    sample_vec = [0];
    [~, n_seg_desire] = size(seg_desire);

    for i = 2:n_seg_desire - 1
        idx = find(l_spline <= seg_desire(i), 1, 'last');
        tmp = sample_t(idx) + (sample_t(idx + 1) - sample_t(idx)) / (l_spline(idx + 1) - l_spline(idx)) * (seg_desire(i) - l_spline(idx));
        sample_vec = [sample_vec, tmp];
    end

    sample_vec = [sample_vec 1];

    XYZ_smooth = fnval(sp_xyz, sample_vec);
    XYZ_smooth = XYZ_smooth';

    % Compute smoothed diameter
    sp_d = spmak(knot, D');
    D_smooth = fnval(sp_d, sample_vec);
    D_smooth = D_smooth';

    % Compute tangent vector
    dsp_xyz = fnder(sp_xyz, 1);
    tangent_vec = fnval(dsp_xyz, sample_vec);
    tangent_vec = tangent_vec';

end
