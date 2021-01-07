#include <cuda_runtime.h>
#include <math.h>
#include "atom_class.h"
#include "bond_class.h"
#include "angle_class.h"
#include "dih_class.h"
#include "read_prmtop.h"
#include "constants.h"

void read_prmtop(char* prmtopFileName, atom& atoms, bond& bonds, angle& angles, dih& dihs, float eps) {

	char line[MAXCHAR];
	char const *FlagSearch = "\%FLAG";
	char const *blank = " ";
	char const *metaDataFlag = "POINTERS";
	char const *chargeFlag = "CHARGE";
	char const *massFlag = "MASS";
	char const *atomTypeIndexFlag = "ATOM_TYPE_INDEX";
	char const *nExcludedAtomsFlag = "NUMBER_EXCLUDED_ATOMS";
	char const *excludedAtomsListFlag = "EXCLUDED_ATOMS_LIST";
	char const *nonBondedParmIndexFlag = "NONBONDED_PARM_INDEX";
	char const *ljACoeffFlag = "LENNARD_JONES_ACOEF";
	char const *ljBCoeffFlag = "LENNARD_JONES_BCOEF";
	char const *bondKFlag = "BOND_FORCE_CONSTANT";
	char const *bondX0Flag = "BOND_EQUIL_VALUE";
	char const *bondnHFlag = "BONDS_WITHOUT_HYDROGEN";
	char const *bondHFlag = "BONDS_INC_HYDROGEN";
	char const *angleKFlag = "ANGLE_FORCE_CONSTANT";
	char const *angleX0Flag = "ANGLE_EQUIL_VALUE";
	char const *anglenHFlag = "ANGLES_WITHOUT_HYDROGEN";
	char const *angleHFlag = "ANGLES_INC_HYDROGEN";
	char const *dihKFlag = "DIHEDRAL_FORCE_CONSTANT";
	char const *dihNFlag = "DIHEDRAL_PERIODICITY";
	char const *dihPFlag = "DIHEDRAL_PHASE";
	char const *dihnHFlag = "DIHEDRALS_WITHOUT_HYDROGEN";
	char const *dihHFlag = "DIHEDRALS_INC_HYDROGEN";
	char const *sceeScaleFactorFlag = "SCEE_SCALE_FACTOR";
	char const *scnbScaleFactorFlag = "SCNB_SCALE_FACTOR";
	char const *atomsPerMoleculeFlag = "ATOMS_PER_MOLECULE";
	char const *solventPointerFlag = "SOLVENT_POINTERS";
	char *flag;
	char *token;
	char *temp;
	int i, nLines;
	int bondCount;
	int angleCount;
	int dihCount;
	int atomCount;
	int parmCount;
	int typeCount;
	int molCount;
	int lineCount;
	int nTypes2;
	int *tempBondArray;
	int *tempAngleArray;
	int *tempDihArray;
	float sqrtEps = sqrt(eps);
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
					temp = fgets(line, MAXCHAR, prmFile);
					/* read meta data section line by line */
					/* line 1: */
					temp = fgets(line, MAXCHAR, prmFile);
					atoms.nAtoms = atoi(strncpy(token,line,8));
					printf("Number of atoms from prmtop file: %d\n", atoms.nAtoms);
					atoms.nTypes = atoi(strncpy(token,line+8,8));
					printf("Number of atom types from prmtop file: %d\n", atoms.nTypes);
					bonds.nBondHs = atoi(strncpy(token,line+16,8));
					printf("Number of bonds containing hydrogens: %d\n", bonds.nBondHs);
					bonds.nBondnHs = atoi(strncpy(token,line+24,8));
					printf("Number of bonds NOT containing hydrogens: %d\n", bonds.nBondnHs);
					bonds.nBonds = bonds.nBondHs + bonds.nBondnHs;
					angles.nAngleHs = atoi(strncpy(token,line+32,8));
					printf("Number of angles containing hydrogens: %d\n", angles.nAngleHs);
					angles.nAnglenHs = atoi(strncpy(token,line+40,8));
					printf("Number of angles NOT containing hydrogens: %d\n", angles.nAnglenHs);
					angles.nAngles = angles.nAngleHs + angles.nAnglenHs;
					dihs.nDihHs = atoi(strncpy(token,line+48,8));
					printf("Number of dihs containing hydrogens: %d\n", dihs.nDihHs);
					dihs.nDihnHs = atoi(strncpy(token,line+56,8));
					printf("Number of dihs NOT containing hydrogens: %d\n", dihs.nDihnHs);
					dihs.nDihs = dihs.nDihHs + dihs.nDihnHs;
					/* line 2: */
					temp = fgets(line, MAXCHAR, prmFile);
					atoms.excludedAtomsListLength = atoi(strncpy(token,line,8));
					printf("Length of excluded atoms list: %d\n", atoms.excludedAtomsListLength);
					bonds.nTypes = atoi(strncpy(token,line+40,8));
					printf("Number of unique bond types: %d\n", bonds.nTypes);
					angles.nTypes = atoi(strncpy(token,line+48,8));
					printf("Number of unique angle types: %d\n", angles.nTypes);
					dihs.nTypes = atoi(strncpy(token,line+56,8));
					printf("Number of unique dih types: %d\n", dihs.nTypes);
					/* line 3: */
					temp = fgets(line, MAXCHAR, prmFile);
					/* line 4: */
					temp = fgets(line, MAXCHAR, prmFile);

					/* allocate arrays */
					atoms.allocate();
					bonds.allocate();
					angles.allocate();
					dihs.allocate();
				} else if (strcmp(flag,chargeFlag) == 0) {
					// read bond k values
					nLines = (int) (atoms.nAtoms + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					atomCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (atomCount < atoms.nAtoms && lineCount < 5) {
							// charge will be fourth element in position float4 array
							atoms.pos_h[atomCount].w = atof(strncpy(token,line+lineCount*16,16))/sqrtEps;
							atomCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,massFlag) == 0) {
					// read bond k values
					nLines = (int) (atoms.nAtoms + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					atomCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (atomCount < atoms.nAtoms && lineCount < 5) {
							atoms.vel_h[atomCount].w = atof(strncpy(token,line+lineCount*16,16));
                                                        atoms.mass_h[atomCount].w = atof(strncpy(token,line+lineCount*16,16));
                                                        // reweight hydrogens
                                                        if (atoms.vel_h[atomCount].w < 2.0) {
                                                                atoms.vel_h[atomCount].w = 12.0;
                                                        }
                                                        atomCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,atomTypeIndexFlag) == 0) {
					// 
					nLines = (int) (atoms.nAtoms + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					atomCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (atomCount < atoms.nAtoms && lineCount < 10) {
							atoms.ityp_h[atomCount] = atoi(strncpy(token,line+lineCount*8,8))-1;// minus one for C zero indexing
							atomCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,nExcludedAtomsFlag) == 0) {
					// 
					nLines = (int) (atoms.nAtoms + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					atomCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (atomCount < atoms.nAtoms && lineCount < 10) {
							if (atomCount ==0) {
								atoms.nExcludedAtoms_h[atomCount] = atoi(strncpy(token,line+lineCount*8,8));
							} else {
								atoms.nExcludedAtoms_h[atomCount] = atoms.nExcludedAtoms_h[atomCount-1] + atoi(strncpy(token,line+lineCount*8,8));
							}
							atomCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,nonBondedParmIndexFlag) == 0) {
					// 
					nTypes2 = atoms.nTypes*atoms.nTypes;
					nLines = (int) (nTypes2 + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					parmCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (parmCount < nTypes2 && lineCount < 10) {
							atoms.nonBondedParmIndex_h[parmCount] = atoi(strncpy(token,line+lineCount*8,8))-1; // subtract one here since C is zero indexed
							parmCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,bondX0Flag) == 0) {
					// read bond k values
					nLines = (int) (bonds.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nTypes && lineCount < 5) {
							//bonds.bondX0Unique[bondCount] = atof(strncpy(token,line+lineCount*16,16));
							bonds.bondParams_h[bondCount].y = atof(strncpy(token,line+lineCount*16,16));
							bondCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,bondKFlag) == 0) {
					// read bond k values
					nLines = (int) (bonds.nTypes + 4) / 5.0;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nTypes && lineCount < 5) {
							//bonds.bondKUnique[bondCount] = atof(strncpy(token,line+lineCount*16,16));
							bonds.bondParams_h[bondCount].x = atof(strncpy(token,line+lineCount*16,16))*2.0; // multiply by two for efficiency in force routine
							bondCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,angleX0Flag) == 0) {
					// read angle k values
					nLines = (int) (angles.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					angleCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (angleCount < angles.nTypes && lineCount < 5) {
							//angles.angleX0Unique[angleCount] = atof(strncpy(token,line+lineCount*16,16));
							angles.angleParams_h[angleCount].y = atof(strncpy(token,line+lineCount*16,16));
							angleCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,angleKFlag) == 0) {
					// read angle k values
					nLines = (int) (angles.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					angleCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (angleCount < angles.nTypes && lineCount < 5) {
							//angles.angleKUnique[angleCount] = atof(strncpy(token,line+lineCount*16,16));
							angles.angleParams_h[angleCount].x = atof(strncpy(token,line+lineCount*16,16))*2; // multiply by two for efficiency in force routine
							angleCount++;
							lineCount++;
						}
					}
				// Dihedral parameters
				} else if (strcmp(flag,dihNFlag) == 0) {
					// read dih periodicity values
					nLines = (int) (dihs.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					dihCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (dihCount < dihs.nTypes && lineCount < 5) {
							dihs.dihParams_h[dihCount].x = atof(strncpy(token,line+lineCount*16,16));
							dihCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,dihKFlag) == 0) {
					// read dih k values
					nLines = (int) (dihs.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					dihCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (dihCount < dihs.nTypes && lineCount < 5) {
							dihs.dihParams_h[dihCount].y = atof(strncpy(token,line+lineCount*16,16));
							dihCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,dihPFlag) == 0) {
					// read dih phase values
					nLines = (int) (dihs.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					dihCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (dihCount < dihs.nTypes && lineCount < 5) {
							dihs.dihParams_h[dihCount].z = atof(strncpy(token,line+lineCount*16,16));
							dihCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,sceeScaleFactorFlag) == 0) {
					// read dih phase values
					nLines = (int) (dihs.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					dihCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (dihCount < dihs.nTypes && lineCount < 5) {
							dihs.scaled14Factors_h[dihCount].x = atof(strncpy(token,line+lineCount*16,16));
							dihCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,scnbScaleFactorFlag) == 0) {
					// read dih phase values
					nLines = (int) (dihs.nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					dihCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (dihCount < dihs.nTypes && lineCount < 5) {
							dihs.scaled14Factors_h[dihCount].y = atof(strncpy(token,line+lineCount*16,16));
							dihCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,ljACoeffFlag) == 0) {
					nTypes2 = atoms.nTypes*(atoms.nTypes+1)/2;
					nLines = (int) (nTypes2 + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					typeCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (typeCount < nTypes2 && lineCount < 5) {
							atoms.lj_h[typeCount].x = atof(strncpy(token,line+lineCount*16,16));
							typeCount++;
							lineCount++;
						}
					}
				} else if (strcmp(flag,ljBCoeffFlag) == 0) {
					nTypes2 = atoms.nTypes*(atoms.nTypes+1)/2;
					nLines = (int) (nTypes2 + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					typeCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (typeCount < nTypes2 && lineCount < 5) {
							atoms.lj_h[typeCount].y = atof(strncpy(token,line+lineCount*16,16));
							typeCount++;
							lineCount++;
						}
					}
				// Bond atoms
				} else if (strcmp(flag,bondHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) (bonds.nBondHs*3 + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					tempBondArray = (int *) malloc(bonds.nBondHs*3*sizeof(int));
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nBondHs*3 && lineCount < 10) {
							tempBondArray[bondCount] = atoi(strncpy(token,line+lineCount*8,8));
							bondCount++;
							lineCount++;
						}
					}
					// parse to bond arrays
					for (i=0;i<bonds.nBondHs;i++) {
						bonds.bondAtoms_h[i].x = tempBondArray[i*3]/3;    // first bond atom
						bonds.bondAtoms_h[i].y = tempBondArray[i*3+1]/3;  // second bond atom
						bonds.bondAtoms_h[i].z = tempBondArray[i*3+2]-1;  // bond type
						//bonds.bondKs_h[i] = bonds.bondKUnique[tempBondArray[i*3+2]-1]*2.0;
						//bonds.bondX0s_h[i] = bonds.bondX0Unique[tempBondArray[i*3+2]-1];
					}
					free(tempBondArray);
				} else if (strcmp(flag,bondnHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) (bonds.nBondnHs*3 + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					bondCount = 0;
					tempBondArray = (int *) malloc(bonds.nBondnHs*3*sizeof(int));
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (bondCount < bonds.nBondnHs*3 && lineCount < 10) {
							tempBondArray[bondCount] = atoi(strncpy(token,line+lineCount*8,8));
							bondCount++;
							lineCount++;
						}
					}
					// parse to bond arrays
					for (i=0;i<bonds.nBondnHs;i++) {
						bonds.bondAtoms_h[(i+bonds.nBondHs)].x = tempBondArray[i*3]/3;
						bonds.bondAtoms_h[(i+bonds.nBondHs)].y = tempBondArray[i*3+1]/3;
						bonds.bondAtoms_h[(i+bonds.nBondHs)].z = tempBondArray[i*3+2]-1;
						//bonds.bondKs_h[(i+bonds.nBondHs)] = bonds.bondKUnique[tempBondArray[i*3+2]-1]*2.0;
						//bonds.bondX0s_h[(i+bonds.nBondHs)] = bonds.bondX0Unique[tempBondArray[i*3+2]-1];
					}
					free(tempBondArray);
				// Angle atoms
				} else if (strcmp(flag,angleHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) (angles.nAngleHs*4 + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					angleCount = 0;
					tempAngleArray = (int *) malloc(angles.nAngleHs*4*sizeof(int));
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (angleCount < angles.nAngleHs*4 && lineCount < 10) {
							tempAngleArray[angleCount] = atoi(strncpy(token,line+lineCount*8,8));
							angleCount++;
							lineCount++;
						}
					}
					// parse to angle arrays
					for (i=0;i<angles.nAngleHs;i++) {
						angles.angleAtoms_h[i].x = tempAngleArray[i*4]/3;
						angles.angleAtoms_h[i].y = tempAngleArray[i*4+1]/3;
						angles.angleAtoms_h[i].z = tempAngleArray[i*4+2]/3;
						angles.angleAtoms_h[i].w = tempAngleArray[i*4+3]-1;
						//angles.angleKs_h[i] = angles.angleKUnique[tempAngleArray[i*4+3]-1]*2.0;
						//angles.angleX0s_h[i] = angles.angleX0Unique[tempAngleArray[i*4+3]-1];
					}
					free(tempAngleArray);
				} else if (strcmp(flag,anglenHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) (angles.nAnglenHs*4 + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					angleCount = 0;
					tempAngleArray = (int *) malloc(angles.nAnglenHs*4*sizeof(int));
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (angleCount < angles.nAnglenHs*4 && lineCount < 10) {
							tempAngleArray[angleCount] = atoi(strncpy(token,line+lineCount*8,8));
							angleCount++;
							lineCount++;
						}
					}
					// parse to angle arrays
					for (i=0;i<angles.nAnglenHs;i++) {
						angles.angleAtoms_h[(i+angles.nAngleHs)].x = tempAngleArray[i*4]/3;
						angles.angleAtoms_h[(i+angles.nAngleHs)].y = tempAngleArray[i*4+1]/3;
						angles.angleAtoms_h[(i+angles.nAngleHs)].z = tempAngleArray[i*4+2]/3;
						angles.angleAtoms_h[(i+angles.nAngleHs)].w = tempAngleArray[i*4+3]-1;
						//angles.angleKs_h[(i+angles.nAngleHs)] = angles.angleKUnique[tempAngleArray[i*4+3]-1]*2.0;
						//angles.angleX0s_h[(i+angles.nAngleHs)] = angles.angleX0Unique[tempAngleArray[i*4+3]-1];
					}
					free(tempAngleArray);
				// Dihedral atoms
				} else if (strcmp(flag,dihHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) (dihs.nDihHs*5 + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					dihCount = 0;
					tempDihArray = (int *) malloc(dihs.nDihHs*5*sizeof(int));
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (dihCount < dihs.nDihHs*5 && lineCount < 10) {
							tempDihArray[dihCount] = atoi(strncpy(token,line+lineCount*8,8)); 
							dihCount++;
							lineCount++;
						}
					}
					// parse to dih arrays
					for (i=0;i<dihs.nDihHs;i++) {
						dihs.dihAtoms_h[i].x = tempDihArray[i*5]/3;
						dihs.dihAtoms_h[i].y = tempDihArray[i*5+1]/3;
						dihs.dihAtoms_h[i].z = tempDihArray[i*5+2]/3;
						dihs.dihAtoms_h[i].w = tempDihArray[i*5+3]/3;
						dihs.dihTypes_h[i] = tempDihArray[i*5+4]-1;
					}
					free(tempDihArray);
				} else if (strcmp(flag,dihnHFlag) == 0) { 
					/* FORMAT 10I8 */
					nLines = (int) (dihs.nDihnHs*5 + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					dihCount = 0;
					tempDihArray = (int *) malloc(dihs.nDihnHs*5*sizeof(int));
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (dihCount < dihs.nDihnHs*5 && lineCount < 10) {
							tempDihArray[dihCount] = atoi(strncpy(token,line+lineCount*8,8));
							dihCount++;
							lineCount++;
						}
					}
					// parse to dih arrays
					for (i=0;i<dihs.nDihnHs;i++) {
						dihs.dihAtoms_h[(i+dihs.nDihHs)].x = tempDihArray[i*5]/3;
						dihs.dihAtoms_h[(i+dihs.nDihHs)].y = tempDihArray[i*5+1]/3;
						dihs.dihAtoms_h[(i+dihs.nDihHs)].z = tempDihArray[i*5+2]/3;
						dihs.dihAtoms_h[(i+dihs.nDihHs)].w = tempDihArray[i*5+3]/3;
						dihs.dihTypes_h[(i+dihs.nDihHs)] = tempDihArray[i*5+4]-1;
					}
					free(tempDihArray);
				} else if (strcmp(flag,excludedAtomsListFlag) == 0) {
					// 
					nLines = (int) (atoms.excludedAtomsListLength + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					atomCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (atomCount < atoms.excludedAtomsListLength && lineCount < 10) {
							atoms.excludedAtomsList_h[atomCount] = atoi(strncpy(token,line+lineCount*8,8))-1;
                                                        //printf("%i \n",atoms.excludedAtomsList_h[atomCount]);
                                                        atomCount++; 
							lineCount++;
						}
					}
				} else if (strcmp(flag,solventPointerFlag) == 0) {
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* Read line with three integers */
					temp = fgets(line, MAXCHAR, prmFile);
					atoms.nMols = atoi(strncpy(token,line+8,8));
					printf("Number of molecules in prmtop file: %d\n", atoms.nMols);
					atoms.allocate_molecule_arrays();
				} else if (strcmp(flag,atomsPerMoleculeFlag) == 0) {
					// 
					nLines = (int) (atoms.nMols + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					molCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (molCount < atoms.nMols && lineCount < 10) {
							atoms.molPointer_h[molCount] = atof(strncpy(token,line+lineCount*8,8));
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




