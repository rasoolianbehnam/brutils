jupyter nbconvert --to html *ipynb
mv -vn *.html  ${PWD/jupyter/html}/
rm *.html