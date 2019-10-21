% n = 5;
% num_lambda = 0;
% for err3D=1:n
%     num_lambda = num_lambda+1;
%     err3D = rand(1);
%     ERR(num_lambda) = err3D;
% end
%  dlmwrite('results.txt', ERR, ' ')

c1= 0;
n = 55;
for ii = 1:n:-1
    for jj = ii+1:n
        c1=c1+1;
    end
end


 b = [1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14];
 fid = fopen('data.txt', 'a'); 
 fprintf('\n');
i = 1;
while i < length(b)
    for j = 0 : 1
        if (i + j) < length(b)
           d = b(i+j);
           fprintf(fid, '%5d \n', d);
        end
    end  
i = i + 4;
end
fclose(fid);

 b = b+[1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14];
 fid = fopen('data.txt', 'a'); 
 fprintf('\n');
i = 1;
while i < length(b)
    for j = 0 : 1
        if (i + j) < length(b)
           d = b(i+j);
           fprintf(fid, '%5d \n', d);
        end
    end  
i = i + 4;
end
fclose(fid);
