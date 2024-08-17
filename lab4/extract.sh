#CODED BY JUNJIE ZHU

fname="NM000207.gb" #change name if required
echo "Reading file: ${fname}"
grep "ORGANISM" ${fname} | sed -e 's/^[ ]*//g' > log

#create and link the static library
gcc -c -o static/info_extract.o info_extract.c
ar rcs lib/libinfo_extract.a static/info_extract.o
gcc -static -o bin/info_extract.static main.c -L./lib -linfo_extract -I./include

#run the static library
bin/info_extract.static >> log

#other information
grep "  exon" ${fname} | sed -e 's/^[ ]*//g' >> log
grep " CDS" ${fname} | sed -e 's/^[ ]*//g' >> log

#output information
echo "File converted successfully!"
