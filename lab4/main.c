#include "info_extract.h"

int main() {
	struct info* sequence;
	sequence=(struct info*)malloc(sizeof(struct info));
	
	//save data to sequence
	sequence->organism = getorg();
	sequence->seq = getseq();
	sequence->mRNA = transcription(sequence->seq);
	sequence->protein = translation();
	
	//convert to fasta
	convert(sequence->seq);

	//output result to log
	printf("DNA sequence:\n %s\n\n",sequence->seq);
	printf("mRNA sequence:\n %s\n\n",sequence->mRNA);
	printf("Protein sequence from CDS:\n %s\n\n",sequence->protein);

	//translate using codon table
	//printf("Converted protein sequence:\n ");
	//char *sqc, *mRNA;
	//sqc = "ATAGGTAATGAT";
	//mRNA = transcription(sqc);
	//trsl_with_codon(mRNA);
	//printf("\n\n");
	return 0;
}
