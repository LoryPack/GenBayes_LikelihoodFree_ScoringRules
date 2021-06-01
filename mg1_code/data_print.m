% Print generated data

fid = fopen('data.txt', 'w');

for i = 1 : 50
   
    fprintf(fid, '%5.2f & %5.2f & %5.2f \\\\ \n ', y_freq(i), y_inter(i), y_rare(i));
    
end

fclose(fid);