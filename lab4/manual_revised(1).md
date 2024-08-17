# lab4 Building static and shared library

### Group 9 Member:朱俊超 朱骏杰 张世杰 江向海

## Manual

##### Name

extract.sh - an executable file that reads and saves data from a genebank file and integrate the extracted information into a fasta file.

##### Synopsis

./extract.sh

##### Description

***extract.sh*** includes functions that reading genebank file, extract information (organism, DNA sequence, exons, CDS regions, etc.) and generate a formal fasta file. 

In the process of running ***extract.sh***, a ***log*** file will be generated, recording organism, DNA, mRNA and protein sequence and other details like exons corresponding to the given gene in genebank file.

***info_extract.c***  provide functions to extract , store and process the data in the genebank file.

​	char* getorg() - get organism info. from genebank file.

​	char* getseq() - get DNA sequence from genebank file.

​	char* transcription(char* seq) - get mRNA sequence from either user input or  a genebank file (default).

​	char* translation() - get protein sequence translated directly from genebank file.

​	void trsl_with_codon(char* mRNA) - a translation function that convert input mRNA sequence into corresponding protein sequence.

​	int convert(char* seq) - convert the info. extracted from a genebank file into a fasta file.

##### Options

If no output is given in ***log*** file, that means the input genebank file may give special base beyond A, T, C and G.

If you want to change the input genebank file, please rename the "fname" and "foutput" in ***info_extract.h*** to corresponding input and output file name.

If you want to convert a user-defined DNA sequence into a corresponding protein sequence, please activate the translation part in ***main.c***.

##### Return Value

Name of the input file

If successfully executed, "file converted successfully!" will be returned.

##### Examples

./extract.sh

: extract information in file defined in ***info_extract.h***

## Perspectives

#### 1.Division of work

朱俊超：1.store the triplet codon table into a C struct; 2.write the function ***void trsl_with_codon()*** , which translate a DNA string into a string of amino acid-- realization of **question 1, 2**; 3.write part of the manual.

朱骏杰：1.store the sequence info. (DNA, mRNA and protein) into a C struct; 2.write the function ***getseq(), transcription() and translation()*** -- realization of **question 3**; 3.build the static library and write the bash script that ensemble all the functions; 4.write part of the manual.

张世杰：1.write a function to store the information into a FASTA file-- realization of ***question 4*** ; 2.write part of the manual; 3.write another file to extract the ORIGIN sequence(independent.c) , failed to take part in the project.

江向海：1. wrote the function **getorg()**;2.wrote part of the manual
