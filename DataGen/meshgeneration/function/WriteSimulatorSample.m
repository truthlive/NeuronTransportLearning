function WriteSimulatorSample(filename, Point, Edge)
[n_point,~]=size(Point);
[n_element,~]=size(Edge);

fid=fopen(filename,'w');
fprintf(fid,'%s\n','# vtk DataFile Version 3.1 ');
fprintf(fid,'%s\n','for LSEConsole');
fprintf(fid,'%s\n','ASCII');
fprintf(fid,'%s\n','DATASET UNSTRUCTURED_GRID');
fprintf(fid,'%s %d %s\n','POINTS',n_point,'FLOAT');
for ii=1:n_point
    fprintf(fid,'%f %f %f\n',Point(ii,1:3));
end
fprintf(fid,'%s %d %d\n','CELLS',n_element,3*n_element);
for  ii=1:n_element
    fprintf(fid,'%d %d %d\n', 2 ,Edge(ii,:));
end
fprintf(fid,'%s %d\n','CELL_TYPES',n_element);
for  ii=1:n_element
    fprintf(fid,'%d\n',3);
end

fclose(fid);

end
