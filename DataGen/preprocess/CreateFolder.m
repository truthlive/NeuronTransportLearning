
% function CreateFolder(test_path)
%     for i = 1:100
%         fld_name = ["..//MLdata//test4_PipeNew", num2str(i, '%04d'); ];
%         fld_name = strjoin(fld_name, '//');
%         mkdir(fld_name);
%     end
% end
for i = 1:100
    skeleton_path = "..//MLdata//skeleton//";
    fld_name = ["..//MLdata//test4_PipeNew", num2str(i, '%04d'); ];
    fld_name = strjoin(fld_name, '//');
    mkdir(fld_name);
end
