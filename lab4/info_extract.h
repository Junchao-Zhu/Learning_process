#ifndef __INFO_EX__
#define __INFO_EX__

#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>

static char* fname="NM000207.gb";
static char* foutput="NM000207.fasta";

struct codon{
        char *amino;
    	char *codon;
};
struct info {
	char* organism;
	char* seq;
	char* mRNA;
	char* protein;
};

char* getorg();
char* getseq();
char* transcription(char* seq);
char* translation();
void trsl_with_codon(char* mRNA);
int convert(char* seq);
#endif

