
num_geo = 100;
num_para_set = 200;
geo_start = 1;
geo_end = 100;
% spline_batch_output = './/spline_src//spline_gen.job'
% metis_batch_output = 'mesh_partition.job'
% nsvms_batch_output = './/nsvms_src//nsvms.job1'
% transport_batch_output = './/transport_src//transport.job1'

spline_batch_output = './/spline_gen.job';
metis_batch_output = 'mesh_partition.job';
nsvms_batch_output = './/nsvms.job4';
transport_batch_output = './/transport.job4';
io_path = '/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/';

fid1 = fopen(spline_batch_output, 'w');
fid2 = fopen(metis_batch_output, 'w');
% fid3 = fopen(nsvms_batch_output, 'w');
% fid4 = fopen(transport_batch_output, 'w');

%% ! Write job file
for idx_job = 1:10
    nNode = 1;
    nProcess = 28;
    fname = ['/pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/job/job', num2str(idx_job, '%04d')];
    fid6 = fopen(fname, 'w');
    fprintf(fid6, '%s\n', '#!/bin/bash');
    fprintf(fid6, '%s\n', '#SBATCH -N ', nNode);
    if(idx_job<2)
    fprintf(fid6, '%s\n', '#SBATCH -p RM-small');
    else
        fprintf(fid6, '%s\n', '#SBATCH -p RM');
    end
    fprintf(fid6, '%s\n', '#SBATCH --ntasks-per-node 28');
    fprintf(fid6, '%s\n', '#SBATCH -t 6:00:00');
    fprintf(fid6, '%s\n', '# echo commands to stdout ');
    fprintf(fid6, '%s\n\n', 'set -x');

    for idx_geo_job = 1:num_geo/10
        idx_geo = idx_geo_job + (idx_job-1)*10;
        path_string = ['-I ', io_path, num2str(idx_geo, '%04d'), '/ '];
        % fn_mesh = ['-m ', 'controlmesh.vtk '];
        fn_parameter = ['-p ', io_path, num2str(idx_geo, '%04d'), '/ ', 'simulation_parameter.txt '];
        % fn_bz = ['-e ', '"bzmeshinfo.txt.epart.', nNode * 28, ' '];
        % fn_velocity = ['-v ', 'initial_velocityfield.txt '];

        
        fprintf(fid6, '%s\n', 'cd /pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/nsvms_src');
        %* Old Commandline
        % line_out = ['mpiexec', ' -np ', '28', ' ./nsvms ', io_path, num2str(idx_geo, '%04d'), '/ 28'];
        %* New Commandline (BC is variable)
        line_out = ['mpiexec', ' -np ', nNode * 28, ' ./nsvms ', path_string, fn_parameter];
        fprintf(fid6, '%s\n\n', line_out);

        fprintf(fid6, '%s\n', 'cd /pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/transport_src');
        % line_out = ['mpiexec', ' -np ', nNode * 28, ' ./transport ', io_path, num2str(idx_geo, '%04d'), '/ 28'];

        %* New Commandline (BC is variable)
        for idx_para_set = 1: num_para_set
            line_out = ['mpiexec', ' -np ', nNode * 28, ' ./transport ', path_string, fn_parameter];
        end

        fprintf(fid6, '%s\n\n', line_out);
        
    end
    fclose(fid6);
end

fname = '/pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/submitAll';
fid6 = fopen(fname,'w');
fprintf(fid6, '%s\n', '#!/bin/bash');
for idx_job = 1:10
    line_out = ['sbatch ', '/pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/job/job', num2str(idx_job, '%04d')];
    fprintf(fid6, '%s\n', line_out);
end
fclose(fid6);

%% ! Write job file BC
for idx_job = 10:10
    nNode = 1;
    nProcess = 28;
    fname = ['/pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/jobBC/job', num2str(idx_job, '%04d')];
    fid6 = fopen(fname, 'w');
    fprintf(fid6, '%s\n', '#!/bin/bash');
    fprintf(fid6, '%s %d\n', '#SBATCH -N ', nNode);

    if (idx_job < 2)
        fprintf(fid6, '%s\n', '#SBATCH -p RM-small');
    else
        fprintf(fid6, '%s\n', '#SBATCH -p RM');
    end

    fprintf(fid6, '%s\n', '#SBATCH --ntasks-per-node 28');
    fprintf(fid6, '%s\n', '#SBATCH -t 10:00:00');
    fprintf(fid6, '%s\n', '# echo commands to stdout ');
    fprintf(fid6, '%s\n\n', 'set -x');
    
    idx_geo = idx_job;
    path_string = ['-I ', io_path, num2str(idx_geo, '%04d'), '/ '];
    % fn_parameter = ['-p ', io_path, num2str(idx_geo, '%04d'), '/ ', 'simulation_parameter.txt '];

    for idx_para_set = 128:128
        
        fn_parameter = ['-p ', io_path, 'simulation_parameter/', num2str(idx_para_set, '%04d'), '.txt', ' -s ', num2str(idx_para_set)];
        % fn_mesh = ['-m ', 'controlmesh.vtk '];
        
        % fn_bz = ['-e ', '"bzmeshinfo.txt.epart.', nNode * 28, ' '];
        % fn_velocity = ['-v ', 'initial_velocityfield.txt '];

        % fprintf(fid6, '%s\n', 'cd /pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/nsvms_src');
        % %* Old Commandline
        % % line_out = ['mpiexec', ' -np ', '28', ' ./nsvms ', io_path, num2str(idx_geo, '%04d'), '/ 28'];
        % %* New Commandline (BC is variable)
        % line_out = ['mpiexec', ' -np ', nNode * 28, ' ./nsvms ', path_string, fn_parameter];
        % fprintf(fid6, '%s\n\n', line_out);

        fprintf(fid6, '%s\n', 'cd /pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/transport_src');
        % line_out = ['mpiexec', ' -np ', nNode * 28, ' ./transport ', io_path, num2str(idx_geo, '%04d'), '/ 28'];

        %* New Commandline (BC is variable)
        line_out = ['mpiexec', ' -np ', num2str(nNode * 28), ' ./transport ', path_string, fn_parameter];
        fprintf(fid6, '%s\n\n', line_out);



    end

    fclose(fid6);
end

fname = '/pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/submitAll_BC';
fid6 = fopen(fname, 'w');
fprintf(fid6, '%s\n', '#!/bin/bash');

for idx_job = 1:300
    line_out = ['sbatch ', '/pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/jobBC/job', num2str(idx_job, '%04d')];
    fprintf(fid6, '%s\n', line_out);
end

fclose(fid6);


%% ! Spline
fprintf(fid1, '%s\n', '#!/bin/bash');
fprintf(fid1, '%s\n', '#SBATCH -N 1');
fprintf(fid1, '%s\n', '#SBATCH -p RM-small');
fprintf(fid1, '%s\n', '#SBATCH --ntasks-per-node 28');
fprintf(fid1, '%s\n', '#SBATCH -t 1:00:00');
fprintf(fid1, '%s\n', '# echo commands to stdout ');
fprintf(fid1, '%s\n', 'set -x');
fprintf(fid1, '%s\n', 'cd /pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/spline_src');
for idx_geo = 1:num_geo
    line_out = [' ./spline', ' ',io_path, num2str(idx_geo, '%04d'), '/'];
    fprintf(fid1, '%s\n',line_out);
end
fclose(fid1);

%% ! Metis
fprintf(fid2, '%s\n', '#!/bin/bash');
fprintf(fid2, '%s\n', '#SBATCH -N 1');
fprintf(fid2, '%s\n', '#SBATCH -p RM-small');
fprintf(fid2, '%s\n', '#SBATCH --ntasks-per-node 28');
fprintf(fid2, '%s\n', '#SBATCH -t 1:00:00');
fprintf(fid2, '%s\n', '# echo commands to stdout ');
fprintf(fid2, '%s\n', 'set -x');

fprintf(fid2, '%s\n', 'module load metis');

for idx_geo = 1:num_geo
    line_out = ['mpmetis', ' ', io_path, num2str(idx_geo, '%04d'), '/bzmeshinfo.txt 28'];
    fprintf(fid2, '%s\n', line_out);
end
fclose(fid2);


% %% ! NSVMS
% fprintf(fid3, '%s\n', '#!/bin/bash');
% fprintf(fid3, '%s\n', '#SBATCH -N 2');
% fprintf(fid3, '%s\n', '#SBATCH -p RM-small');
% fprintf(fid3, '%s\n', '#SBATCH --ntasks-per-node 56');
% fprintf(fid3, '%s\n', '#SBATCH -t 8:00:00');
% fprintf(fid3, '%s\n', '# echo commands to stdout ');
% fprintf(fid3, '%s\n', 'set -x');
% fprintf(fid3, '%s\n', 'cd /pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/nsvms_src');

% for idx_geo = geo_start:geo_end
%     line_out = ['mpiexec', ' -np ', '56', ' ./nsvms ',io_path, num2str(idx_geo, '%04d'), '/ 56'];
%     fprintf(fid3, '%s\n', line_out);
% end
% fclose(fid3);

% %% ! Transport
% fprintf(fid4, '%s\n', '#!/bin/bash');
% fprintf(fid4, '%s\n', '#SBATCH -N 2');
% fprintf(fid4, '%s\n', '#SBATCH -p RM-small');
% fprintf(fid4, '%s\n', '#SBATCH --ntasks-per-node 56');
% fprintf(fid4, '%s\n', '#SBATCH -t 8:00:00');
% fprintf(fid4, '%s\n', '# echo commands to stdout ');
% fprintf(fid4, '%s\n', 'set -x');
% fprintf(fid4, '%s\n', 'cd /pylon5/eg560mp/angranl/NeuronMachineLearning/DataGen/transport_src');

% for idx_geo = geo_start:geo_end
%     line_out = ['mpiexec', ' -np ', '56', ' ./transport ', io_path, num2str(idx_geo, '%04d'), '/ 56'];
%     fprintf(fid4, '%s\n', line_out);
% end
% fclose(fid4);

%% ! Write Simulation Parameter file
% for idx_geo = 1:num_geo
%     fname = [io_path, num2str(idx_geo, '%04d'), '/simulation_parameter.txt'];
%     fid5 = fopen(fname,'w');
%     fprintf(fid5, '%s\n', 'D 1.0');
%     fprintf(fid5, '%s\n', 'vplus 0.10');
%     fprintf(fid5, '%s\n', 'vminus -0.0');
%     fprintf(fid5, '%s\n', 'kplus 1.0');
%     fprintf(fid5, '%s\n', 'kminus 0.0');
%     fprintf(fid5, '%s\n', 'k''plus 0.5');
%     fprintf(fid5, '%s\n', 'k''minus 0.0');
%     fprintf(fid5, '%s\n', 'dt 0.1');
%     fprintf(fid5, '%s\n', 'nstep 100');
%     fprintf(fid5, '%s\n', 'N0bc 1.0');
%     fprintf(fid5, '%s\n', 'Nplusbc 2.0');
%     fprintf(fid5, '%s\n', 'Nminusbc 0.0');

%     fclose(fid5);
% end

%% ! Write Simulation Parameter file different BC

% for idx_para_set = 1:num_para_set
%     N0bc = idx_para_set * 0.05/3.;
%     Nplusbc = idx_para_set * 0.05/3. * 2.;
%     fname = [io_path, '/simulation_parameter/', num2str(idx_para_set, '%04d'), '.txt'];
%     fid5 = fopen(fname, 'w');
%     fprintf(fid5, '%s\n', 'D 1.0');
%     fprintf(fid5, '%s\n', 'vplus 0.10');
%     fprintf(fid5, '%s\n', 'vminus -0.0');
%     fprintf(fid5, '%s\n', 'kplus 1.0');
%     fprintf(fid5, '%s\n', 'kminus 0.0');
%     fprintf(fid5, '%s\n', 'k''plus 0.5');
%     fprintf(fid5, '%s\n', 'k''minus 0.0');
%     fprintf(fid5, '%s\n', 'dt 0.1');
%     fprintf(fid5, '%s\n', 'nstep 100');
%     fprintf(fid5, '%s%f\n', 'N0bc ', N0bc);
%     fprintf(fid5, '%s%f\n', 'Nplusbc ', Nplusbc);
%     fprintf(fid5, '%s\n', 'Nminusbc 0.0');
%     fclose(fid5);
% end

fclose('all');

