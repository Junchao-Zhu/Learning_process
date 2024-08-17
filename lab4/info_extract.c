#include "info_extract.h"

char* getorg() {
	FILE* log;
	char un[10],org1[10],org2[10];

	log = fopen("log","r");
	fscanf(log,"%s %s %s",un,org1,org2);
	char* org = (char *) malloc(strlen(org1) + strlen(org2));
	strcpy(org, org1);
	strcat(org, " ");
	strcat(org, org2);
	return org;
}	

char* getseq() {
	FILE* gb;
        static char seq[1000]; //save sequence
        char readline[250]; //read current line
	bool reach=false; //decide if sequence
	int i=0,len=0; //sequence length
	
	gb = fopen("NM000207.gb","r");
	while (fgets(readline, sizeof(readline), gb)) {
		if (reach==true) {
			for (i=0; i < strlen(readline); i++){
				if (readline[i]=='a' || readline[i]=='t' || readline[i]=='c' || readline[i]=='g')
					seq[len++] = readline[i] - ('a' - 'A');
			}
		}
		else{
			if (readline[0]=='O' && readline[1]=='R') reach=true;
		}
	}
	seq[len] = '\0';
	fclose(gb);
	return seq;
}

char* transcription(char* seq) {
	char* mRNA = (char*)malloc(strlen(seq) * sizeof(char));
	int i=0;
	for (i; i < strlen(seq); i++) {
		if (seq[i]=='A') mRNA[i]='U';
        	else if (seq[i]=='T') mRNA[i]='A';
        	else if (seq[i]=='C') mRNA[i]='G';
        	else if (seq[i]=='G') mRNA[i]='C';
	}
	return mRNA;
}

char* translation() {
        FILE* gb;
        static char protein[1000]; //save protein sequence
        char readline[250]; //read current line
        bool reach=false; //switch reach state
        int k,l=0,len=0; //protein sequence length
        char* start="/translation"; //start signal
        char* end="sig_peptide"; //end signal

        gb = fopen(fname,"r");
        while (fgets(readline, sizeof(readline), gb)) {
                if (strstr(readline,end) != NULL) break;
                if (reach==true) {
                        for (k = 0; k < strlen(readline); k++) {
                                if (readline[k] >= 'A' && readline[k] <= 'Z')
                                        protein[len++] = readline[k];
                        }
                }
                else {
                        if (strstr(readline,start) != NULL) {
                                reach=true;
                                for (k = 0; k < strlen(readline); k++) {
                                        if (readline[k] >= 'A' && readline[k] <= 'Z')
                                                protein[len++] = readline[k];
                                }
                        }
                }
        }
        protein[len] = '\0';
        fclose(gb);
//      printf("%d\n",strlen(protein));
        return protein; 
}

void trsl_with_codon(char* mRNA){
        //build coden structure
	struct codon aa[] = {{"F","UUUUUC"},
          {"L","UUAUUGCUUCUCCUACUG"},
          {"I","AUUAUCAUA"},
          {"M","AUG"},
          {"V","GUUGUCGUAGUG"},
          {"S","UCUUCCUCAUCGAGUAGC"},
          {"P","CCUCCCCCACCG"},
          {"T","ACUACCACAACG"},
          {"A","GCUGCCGCAGCG"},
          {"Y","UAUUAC"},
          {"H","CAUCAC"},
          {"Q","CAACAG"},
          {"N","AAUAAC"},
          {"K","AAAAAG"},
          {"D","GAUGAC"},
          {"E","GAAGAG"},
          {"C","UGUUGC"},
          {"W","UGG"},
          {"R","CGUCGCCGACGGAGAAGG"},
          {"G","GGUGGCGGAGGG"}},*ps;
	//sequence check
	int check;
    	for(check=0;check<strlen(mRNA);check++){
        	if(mRNA[check] != 'A' && mRNA[check] != 'U' && mRNA[check] != 'C' && mRNA[check] != 'G' ){
            		printf("Unrecognized base!");
            		check = 0;
            		break;
        	}
    	}
    	if(check != 0)
        	if(strlen(mRNA)%3 != 0){
            		printf("Wrong sequence number!!!");
        	}
        	else{
            		int i, j, k = 0;
            		int p, q = 0;
        	//get sequence coden dividedly
        		while(k<strlen(mRNA)){
        			char *divided_coden = (char*)malloc(3 * sizeof(char));
        			for(i=0;i<3;i++){
            				if (mRNA[k]=='A') divided_coden[i]='A';
        				else if (mRNA[k]=='U') divided_coden[i]='U';
        				else if (mRNA[k]=='C') divided_coden[i]='C';
        				else if (mRNA[k]=='G') divided_coden[i]='G';
            				k = k+1;
        			}

		//traverse codon table
        			for(p=0;p<20;p++){
            				char coden_cut[20];
            				strcpy(coden_cut, aa[p].codon);
            				int h = 0;
            				while(h<strlen(coden_cut)){
                				char *trans_coden = (char*)malloc(3 * sizeof(char));
                				for(q=0;q<3;q++){
                    					if (coden_cut[h]=='A') trans_coden[q]='A';
        	        				else if (coden_cut[h]=='U') trans_coden[q]='U';
        	        				else if (coden_cut[h]=='C') trans_coden[q]='C';
        	        				else if (coden_cut[h]=='G') trans_coden[q]='G';
                    					h = h+1;
            					}
            					if(strcmp(trans_coden,divided_coden) == 0 ){
                    					char am[1];
                    					strcpy(am, aa[p].amino);
                    					printf("%s",am);
                				}
            				}	      
        			}
    			}
		}
	
}

int convert(char* seq){
	FILE *gb, *fasta;
        char a[10],b[100];
        char c,d;
	gb = fopen(fname,"r");
	fasta=fopen(foutput,"wb");
	
	fseek(gb,0L,0);
	fputc('>',fasta);
	do{
		fgets(b,8,gb);
	}while(strcmp(b,"VERSION")!=0);
	do{
		d=fgetc(gb);
		if (d!=' '&& d!='\n'){
		fputc(d,fasta);}
	}while(d!='\n');
	fseek(gb,0L,0);
	fputc('\t',fasta);
	do {
		fgets(a,11,gb);
	}while(strcmp(a,"DEFINITION")!=0);
	do{
		c=fgetc(gb);
		if (c!='\n'){
		fputc(c,fasta);}
	}while(c!='\n');
	fputs("\n",fasta);
	fputs(seq,fasta);
	fclose(gb);
	fclose(fasta);
}

