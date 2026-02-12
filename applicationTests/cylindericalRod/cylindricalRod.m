load cylindricalRod.mat
sys = sparss(A,B,C,D,E);
size(sys)
R = reducespec(sys,"balanced");
R = process(R)
rsys = getrom(R,MaxError=1e-6,Method="truncate");