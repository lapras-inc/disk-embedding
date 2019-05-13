
basename=$1

for fn in $basename*; do
  sed -e "s/\(.*\)	\(.*\)/\2	\1/" $fn > "r_$fn"
done

