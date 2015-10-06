data=`ls data/ -tr| head -300000`

for f in $data; do
  echo $f
  rm data/$f
done
