function [M2Smapping] = MeshToSimulator(fname_MeshMapping,fname_SimMapping)

    MeshMapping = dlmread(fname_MeshMapping);
    SimMapping = dlmread(fname_SimMapping);

    n_mesh = size(MeshMapping,2);
    idx_mesh =1:n_mesh;


    LUT_mesh(MeshMapping + 1) = idx_mesh;
    M2Smapping = LUT_mesh(SimMapping + 1);
    
    
end

