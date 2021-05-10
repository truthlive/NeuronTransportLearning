function [Edge] = MeshToEdge(Mesh)

[ne,np]=size(Mesh);

if np == 8
Edge_dup = [Mesh(:,[1,2]);Mesh(:,[2,3]);Mesh(:,[3,4]);Mesh(:,[4,1]);
    Mesh(:,[5,6]);Mesh(:,[6,7]);Mesh(:,[7,8]);Mesh(:,[8,5]);
    Mesh(:,[1,5]);Mesh(:,[2,6]);Mesh(:,[3,7]);Mesh(:,[4,8]);];
elseif np == 4
    Edge_dup = [Mesh(:,[1,2]);Mesh(:,[2,3]);Mesh(:,[3,4]);Mesh(:,[4,1])];
end

Edge_sort = sort(Edge_dup,2);
[Edge,~,~] = unique(Edge_sort,'rows'); % determine uniqueness of edges

end

