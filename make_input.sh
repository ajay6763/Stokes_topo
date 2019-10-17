data=post_processing_output.dat
read -p 'Enter profile length: ' profile_len
# For gridding bounds
bounds="-R0/$profile_len/0/400"
awk '{print $1, -$2, $3}' $data | xyz2grd $bounds -I5/5 -Gtemp.grd  -V
 blockmedian  -I100m/100m
awk '{print $1, -$2, $3}' $data | blockmedian  -I5/5 $bounds  -Gtemp.grd  -V
awk '{print $1, -$2, $3}' $data | nearneighbor  -I5/5 $bounds -S10 -N4 -F  -Gtemp.grd  -V
grd2xyz temp.grd > Temperature.xyz

awk '{print $1, -$2, $7}' $data | xyz2grd $bounds -I5/5 -Gtemp.grd  -V
grd2xyz temp.grd > Density.xyz



