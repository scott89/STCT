a=`ls benchmark_res/*.mat`

for x in $a; do
  #echo $x
  name=`echo $x | sed "s/base/_base/g"`
  echo $name
  mv $x $name
done
