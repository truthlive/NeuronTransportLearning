% idx_except = [5,6,7,8,14,15,16,23,24,63,44,48,50,56,57,62,66,67,72,75,81,82,85,95,96,97];
% for i=1:10
%     if(find(idx_except==i))
%         fprintf("Got it!\n");
%         continue;
%     end
%     fprintf("%d\n",i);
% end

fname_MeshMapping = '..//..//MLdata//NMO_66748//MeshComponent//Pipe//0002//mesh_S2Dmapping.txt';
fname_SimMapping = '..//..//MLdata//NMO_66748//MeshComponent//Pipe//0002//simulator_S2Dmapping.txt';
M2Smapping = [];
M2Smapping = MeshToSimulator(fname_MeshMapping,fname_SimMapping);

fname_M2Smapping =  '..//..//MLdata//NMO_66748//MeshComponent//Pipe//0002//M2Smapping.txt';
dlmwrite(fname_M2Smapping,M2Smapping-1,'Delimiter', ',', 'precision', '%d')

% MeshMapping = dlmread(fname_MeshMapping);
% SimMapping = dlmread(fname_SimMapping);
% 
% n_mesh = size(MeshMapping,2);
% idx_mesh =1:n_mesh;
% 
% 
% LUT_mesh(MeshMapping) = idx_mesh;
% mapping_MeshToSim = LUT_mesh(SimMapping);
