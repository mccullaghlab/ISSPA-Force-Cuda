
#include "atom_class.h"
#include "bond_class.h"
#include "read_prmtop.h"
#include "constants.h"

void read_prmtop(char* prmtopFileName, atom& atoms, bond& bonds) {

	char line[MAXCHAR];
	char const *FlagSearch = "\%FLAG";
	char const *blank = " ";
	char const *metaDataFlag = "POINTERS";
	char const *chargeFlag = "CHARGE";
	char const *massFlag = "MASS";
	char const *atomTypeIndexFlag = "ATOM_TYPE_INDEX";
	char const *nExcludedAtomsFlag = "NUMBER_EXCLUDED_ATOMS";
	char const *nonBondIndexFlag = "NONBONDED_PARM_INDEX";
	char const *bondKFlag = "BOND_FORCE_CONSTANT";
	char const *bondX0Flag = "BOND_EQUIL_VALUE";
	char const *bondnHFlag = "BONDS_WITHOUT_HYDROGEN";
	char const *bondHFlag = "BONDS_INC_HYDROGEN";
	char const *atomsPerMoleculeFlag = "ATOMS_PER_MOLECULE";
	char const *solventPointerFlag = "SOLVENT_POINTERS";
	char *flag;
	char *token;
	int i, nLines;
	int bondCount;
	int atomCount;
	int molCount;
	int lineCount;
	int *tempBondArray;
	FILE *prmFile = fopen(prmtopFileName, "r");

	if ( prmFile != NULL) {
		while (fgets(line, MAXCHAR, prmFile) != NULL) {
			if (strncmp(line,FlagSearch,5)==0) {
				token = strtok(line, blank);
				flag = strtok(NULL, blank);
				// printf("FLAG: %s\n", flag); // for DEBUG
				if (strcmp(flag,metaDataFlag) == 0) {
					// read meta data
					printf("Reading system metadata from prmtop file\n");
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* read meta data section line by line */
					/* line 1: */
					fgets(line, MAXCHAR, prmFile);
					atoms.nAtoms = atoi(strncpy(token,line,8));
					printf("Number of atoms from prmtop file: %d\n", atoms.nAtoms);
					atoms.nAtomTypes = atoi(strncpy(token,line+8,8));
					printf("Number of atom types from prmtop file: %d\n", atoms.nAtomTypes);
					bonds.nBondHs = atoi(strncpy(token,line+16,8));
					printf("Number of bonds containing hydrogens: %d\n", bonds.nBondHs);
					bonds.nBondnHs = atoi(strncpy(token,line+24,8));
					printf("Number of bonds NOT containing hydrogens: %d\n", bonds.nBondnHs);
					bonds.nBonds = bonds.nBondHs + bonds.nBondnHs;
					/* line 2: */
					fgets(line, MAXCHAR, prmFile);
					bonds.nTypes = atoi(strncpy(token,line+40,8));
					printf("Number of unique bond types: %d\n", bonds.nTypes);
					/* line 3: */
					fgets(line, MAXCHAR, prmFile);
					/* line 4: */
					fgets(line, MAXCHAR, prmFile);

					/* allocate arrays */
					atoms.allocate();
					bonds.allocate();
				} else if (strcmp(flag,massFlag) == 0) {
					// read bond k values
					nLines = (int) atoms.nAtoms / 5.0 + 1;
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					atomCount = 0;
					for (i=0;i<nLines;i++) {
						fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (atomCount < atoms.nAtoms && lineCount < 5) {
							atoms.mass_h[atomCount] = atof(strncpy(token,line+lineCount*16,16));
							atomCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,bondHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) bonds.nBondHs*3 / 10.0 + 1;
					//DEBUG printf("number of lines to read for bonds with hydrogens: %d\n", nLines);
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					tempBondArray = (int *) malloc(bonds.nBondHs*3*sizeof(int));
					for (i=0;i<nLines;i++) {
						fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nBondHs*3 && lineCount < 10) {
							tempBondArray[bondCount] = atoi(strncpy(token,line+lineCount*8,8));
							bondCount++;
							lineCount++;
						}
					}
					// parse to bond arrays
					for (i=0;i<bonds.nBondHs;i++) {
						bonds.bondAtoms_h[i*2] = tempBondArray[i*3];
						bonds.bondAtoms_h[i*2+1] = tempBondArray[i*3+1];
						bonds.bondKs_h[i] = bonds.bondKUnique[tempBondArray[i*3+2]-1]*2.0;
						bonds.bondX0s_h[i] = bonds.bondX0Unique[tempBondArray[i*3+2]-1];
						//printf("Bond %d is between atoms %d and %d with force constant %f and eq value of %f\n", i+1, bonds.bondAtoms[i*2], bonds.bondAtoms[i*2+1],bonds.bondKs[i],bonds.bondX0s[i]);
					}
					free(tempBondArray);
				} else if (strcmp(flag,bondnHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) bonds.nBondnHs*3 / 10.0 + 1;
					//printf("number of lines to read for bonds without hydrogens: %d\n", nLines);
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					tempBondArray = (int *) malloc(bonds.nBondnHs*3*sizeof(int));
					for (i=0;i<nLines;i++) {
						fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nBondnHs*3 && lineCount < 10) {
							tempBondArray[bondCount] = atoi(strncpy(token,line+lineCount*8,8));
							bondCount++;
							lineCount++;
						}
					}
					// parse to bond arrays
					for (i=0;i<bonds.nBondnHs;i++) {
						bonds.bondAtoms_h[(i+bonds.nBondHs)*2] = tempBondArray[i*3];
						bonds.bondAtoms_h[(i+bonds.nBondHs)*2+1] = tempBondArray[i*3+1];
						bonds.bondKs_h[(i+bonds.nBondHs)] = bonds.bondKUnique[tempBondArray[i*3+2]-1]*2.0;
						bonds.bondX0s_h[(i+bonds.nBondHs)] = bonds.bondX0Unique[tempBondArray[i*3+2]-1];
						//printf("Bond %d is between atoms %d and %d with force constant %f and eq value of %f\n", (i+bonds.nBondHs)+1, bonds.bondAtoms[(i+bonds.nBondHs)*2], bonds.bondAtoms[(i+bonds.nBondHs)*2+1],bonds.bondKs[(i+bonds.nBondHs)],bonds.bondX0s[(i+bonds.nBondHs)]);
					}
					free(tempBondArray);
				} else if (strcmp(flag,bondX0Flag) == 0) {
					// read bond k values
					nLines = (int) bonds.nTypes / 5.0 + 1;
					//printf("number of lines to read for bond equilibrium value: %d\n", nLines);
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					for (i=0;i<nLines;i++) {
						fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nTypes && lineCount < 5) {
							bonds.bondX0Unique[bondCount] = atof(strncpy(token,line+lineCount*16,16));
							bondCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,bondKFlag) == 0) {
					// read bond k values
					nLines = (int) bonds.nTypes / 5.0 + 1;
					//printf("number of lines to read for bond force constant: %d\n", nLines);
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					for (i=0;i<nLines;i++) {
						fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nTypes && lineCount < 5) {
							bonds.bondKUnique[bondCount] = atof(strncpy(token,line+lineCount*16,16));
							bondCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,solventPointerFlag) == 0) {
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* Read line with three integers */
					fgets(line, MAXCHAR, prmFile);
					atoms.nMols = atoi(strncpy(token,line+lineCount*8,8));
					printf("Number of molecules in prmtop file: %d\n", atoms.nMols);
					atoms.allocate_molecule_arrays();
				} else if (strcmp(flag,atomsPerMoleculeFlag) == 0) {
					// read bond k values
					nLines = (int) atoms.nMols / 10.0 + 1;
					/* skip format line */
					fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					molCount = 0;
					for (i=0;i<nLines;i++) {
						fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (molCount < atoms.nMols && lineCount < 10) {
							atoms.molPointer_h[molCount] = atof(strncpy(token,line+lineCount*8,8));
							printf("%d %d\n", molCount + 1, atoms.molPointer_h[molCount]);
							molCount++;
							lineCount++;
						}
					}
				}			
			}
		}
		fclose( prmFile );
	}

}




