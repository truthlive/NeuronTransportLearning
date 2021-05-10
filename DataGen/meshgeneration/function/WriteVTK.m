function WriteVTK(filename, Point, Element, PointLabel, ElementLabel)
%myFun - Description
%
% Syntax: output = WriteSWC(filename, Mat_swc)
%
% Long description

[n_point,~]=size(Point);
[n_element,~]=size(Element);

fid=fopen(filename,'w');
fprintf(fid,'%s\n','# vtk DataFile Version 3.1 ');
fprintf(fid,'%s\n','for LSEConsole');
fprintf(fid,'%s\n','ASCII');
fprintf(fid,'%s\n','DATASET UNSTRUCTURED_GRID');
fprintf(fid,'%s %d %s\n','POINTS',n_point,'FLOAT');
for ii=1:n_point
    fprintf(fid,'%f %f %f\n',Point(ii,1:3));
end
fprintf(fid,'%s %d %d\n','CELLS',n_element,9*n_element);
for  ii=1:n_element
    fprintf(fid,'%d %d %d %d %d %d %d %d %d\n', 8 ,Element(ii,:));
end
fprintf(fid,'%s %d\n','CELL_TYPES',n_element);
for  ii=1:n_element
    fprintf(fid,'%d\n',12);
end
if(~isempty(PointLabel))
    fprintf(fid,'%s %d\n','POINT_DATA',n_point);
    fprintf(fid,'%s\n','SCALARS label float 1');
    fprintf(fid,'%s\n','LOOKUP_TABLE default');
    for  ii=1:n_point
        fprintf(fid,'%d\n',PointLabel(ii,:));
    end
end
if(~isempty(ElementLabel))
    fprintf(fid,'%s %d\n','CELL_DATA',n_element);
    fprintf(fid,'%s\n','SCALARS Idx_Simulator int 1');
    fprintf(fid,'%s\n','LOOKUP_TABLE default');
    for  ii=1:n_element
        fprintf(fid,'%d\n',ElementLabel(ii,:));
    end
end

fclose(fid);

end
