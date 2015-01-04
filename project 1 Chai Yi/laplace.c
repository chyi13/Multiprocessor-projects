/*
author: Chai Yi 
email: chyi13@student.bth.se
date: 04/16/2014
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define MAX_SIZE 2048
#define DEBUG 0
#define START 1	// send message tag 
#define WORK 2	// slave work tag
#define EXIT 3	// work message tag
#define GET 4

static double matrixA[MAX_SIZE+2][MAX_SIZE+2]; // (+2) - boundary elements
static double matrixB[MAX_SIZE+2][MAX_SIZE+2];	// temp matrix

static int maxnum = 15;				// max number of element
static int real_size = 2048;		// matrix size without boundary
static int option = 0;				// fast = 0  rand = 1
static double difflimit = 0.02048;	// acceptance value
static double w = 0.5;
static int print_r = 0;		// print result
static char* file_name = NULL;		// file name

static void read_options(int argc, char** argv);	// read options from user
static void init_matrix(void);					// initialize matrix with options
static void print_matrix(void);					// print matrix

int main(int argc, char** argv)
{	
	int myrank, nproc;			// MPI part
	int mtype;
	int src, dest;
	int row_size;
	int row,col;
	int message;
	int count = 0;
	double sum;
	double start_time, end_time;
	MPI_Status status;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if (myrank == 0)	// master
	{
		read_options(argc,argv);
		init_matrix();		// initialize matrix

		start_time = MPI_Wtime();

		row_size = real_size/nproc;		// large row size
		
		// send works
		mtype = START;
		for (dest = 1; dest<nproc; dest++)
		{
			row = dest*row_size+1;
			MPI_Send(0,0,MPI_INT,dest,mtype,MPI_COMM_WORLD);		// start message
			MPI_Send(&real_size,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&row,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&row_size,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&matrixA[row-1][0],(row_size+2)*(MAX_SIZE+2),MPI_DOUBLE,dest,mtype,MPI_COMM_WORLD);
		}

		double prev_maxi = 0;
		while(1)
		{	
			count++;
			// master work
			int i,j;
			for (i = 1; i<=row_size; i++)
			{
				for (j = 1; j<=real_size; j++)
				{
					if ((i+j)%2 == (count%2))
					{
						matrixA[i][j] = (1-w)*matrixA[i][j]
							+ w*(matrixA[i-1][j]+matrixA[i+1][j]+matrixA[i][j-1]+matrixA[i][j+1])/4;
					}
				}
			}

			// copy result
			double maxi = -100000;
			for (i = 1; i<=row_size; i++)
			{
				sum = 0;
				for (j = 1; j<=real_size; j++)
				{
					sum += matrixA[i][j];
				}
				if (maxi<sum)
					maxi = sum;
			}
			
			// receive
			double temp_maxi = -100000;
			for (src =1; src<nproc; src++)
			{
				mtype = WORK;
				row = src*row_size+1;
				MPI_Recv(&temp_maxi, 1, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&matrixA[row][0], MAX_SIZE+2, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD,&status);
				MPI_Recv(&matrixA[row+row_size-1][0], MAX_SIZE+2, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD,&status);

				if (maxi<temp_maxi)
					maxi = temp_maxi;
			}
			if (fabs(maxi-prev_maxi)<=difflimit)
			{
				break;
			}
			prev_maxi = maxi;

			// send
			for (dest = 1; dest<nproc; dest++)
			{
				mtype = WORK;
				row = dest*row_size+1;
				MPI_Send(0,0,MPI_INT,dest,mtype,MPI_COMM_WORLD);		// start message
				MPI_Send(&matrixA[row-1][0],MAX_SIZE+2,MPI_DOUBLE,dest,mtype,MPI_COMM_WORLD);
				MPI_Send(&matrixA[row+row_size][0],MAX_SIZE+2,MPI_DOUBLE,dest,mtype,MPI_COMM_WORLD);
			}
		}
		
		mtype = EXIT;		// end all slave node
		for (dest = 1; dest < nproc; dest++)
		{
			MPI_Send(0, 0, MPI_INT, dest, mtype, MPI_COMM_WORLD);
		}

		mtype = GET;		// get final result
		for (src = 1; src < nproc; src++)
		{
			row = src*row_size+1;
			MPI_Recv(&matrixA[row][0],(MAX_SIZE+2)*row_size,MPI_DOUBLE,src,mtype,MPI_COMM_WORLD,&status);
		}

		end_time = MPI_Wtime();
		if (print_r)
			print_matrix();
		printf("Execution time on %2d nodes: %f\n", nproc, end_time-start_time);
		printf("difflimit: %.7lf\ncount: %d\n",difflimit,count);
	}
	else
	{
		count = 0;
		while(1)
		{
			count++;
			// receive
			MPI_Recv(&message, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == START)		// 1.start 
			{
				mtype = START;
				MPI_Recv(&real_size, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&row, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&row_size, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&matrixA[row-1][0], (row_size+2)*(MAX_SIZE+2), MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);

				int i,j;
				for (i = row; i<row+row_size; i++)
				{
					for (j = 1; j<=real_size; j++)
					{
						if ((i+j)%2 == (count%2))
						{
							matrixA[i][j] = (1-w)*matrixA[i][j]
							+ w*(matrixA[i-1][j]+matrixA[i+1][j]+matrixA[i][j-1]+matrixA[i][j+1])/4;
						}
					}
				}
				// copy 
				double maxi = -100000;
				for (i = row; i<row+row_size; i++)
				{
					sum = 0;
					for (j = 1; j<=real_size; j++)
					{
						sum += matrixA[i][j];
					}
					if (maxi<sum)
						maxi = sum;
				}
				// send back
				mtype = WORK;
				MPI_Send(&maxi, 1, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
				MPI_Send(&matrixA[row][0], MAX_SIZE+2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
				MPI_Send(&matrixA[row+row_size-1][0], MAX_SIZE+2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);

			}
			else if (status.MPI_TAG == WORK)	// 2. work
			{
				mtype = WORK;
				MPI_Recv(&matrixA[row-1][0],MAX_SIZE+2,MPI_DOUBLE,0,mtype,MPI_COMM_WORLD,&status);
				MPI_Recv(&matrixA[row+row_size][0],MAX_SIZE+2,MPI_DOUBLE,0,mtype,MPI_COMM_WORLD,&status);
				// work
				int i,j;
				for (i = row; i<row+row_size; i++)
				{
					for (j = 1; j<=real_size; j++)
					{
						if ((i+j)%2 == (count%2))
						{
							matrixA[i][j] = (1-w)*matrixA[i][j]
							+ w*(matrixA[i-1][j]+matrixA[i+1][j]+matrixA[i][j-1]+matrixA[i][j+1])/4;
						}
					}
				}
				// copy
				double maxi = -100000;
				for (i = row; i<row+row_size; i++)
				{
					sum = 0;
					for (j = 1; j<=real_size; j++)
					{
						sum += matrixA[i][j];
					}
					if (maxi<sum)
						maxi = sum;
				}
				// send
				mtype = WORK;
				MPI_Send(&maxi, 1, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
				MPI_Send(&matrixA[row][0], MAX_SIZE+2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
				MPI_Send(&matrixA[row+row_size-1][0], MAX_SIZE+2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}
			else if (status.MPI_TAG == EXIT)		// 3.exit 
			{
				mtype = GET;
				MPI_Send(&matrixA[row][0], row_size*(MAX_SIZE+2), MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
				break;
			}
		}

	}

	MPI_Finalize();

	return 0;
}

static void init_matrix(void)
{
	srand(time(NULL));
	FILE* file;
	if (file_name!=NULL)
	{
		file = fopen(file_name,"r");
		if (file)
		{
			option = 2;
			fscanf(file,"%d",&real_size);	// read matrix size
		}
		else
			exit(0);
	}
	else
	{
		option = 0;
		real_size = MAX_SIZE;
	}
	// initialize
	int i,j;
	int dmmy = 0;
	for (i = 1; i < real_size+1; i++)
	{
		dmmy++;
		for (j = 1; j < real_size+1; j++) 
		{
			dmmy++;
			if (option == 0)
			{
				if ((dmmy%2)== 0)
					matrixA[i][j] = 1.0;
				else
					matrixA[i][j] = 5.0;
			}
			
			else if (option == 1)
				matrixA[i][j] = (rand() % maxnum) + 1.0;
			
			else if (option == 2)
				fscanf(file,"%lf",&matrixA[i][j]);
		}
	}
	if (option == 2)			// close file
		fclose(file);

	// Set the border to the same values as the outermost rows/columns
	// fix the corners
	matrixA[0][0] = matrixA[1][1];
	matrixA[0][real_size+1] = matrixA[1][real_size];
	matrixA[real_size+1][0] = matrixA[real_size][1];
	matrixA[real_size+1][real_size+1] = matrixA[real_size][real_size];
	// fix the top and bottom rows
	for (i = 1; i < real_size+1; i++)
	{
		matrixA[0][i] = matrixA[1][i];
		matrixA[real_size+1][i] = matrixA[real_size][i];
	}
	// fix the left and right columns
	for (i = 1; i < real_size+1; i++)
	{
		matrixA[i][0] = matrixA[i][1];
		matrixA[i][real_size+1] = matrixA[i][real_size];
	}

	if (DEBUG)
		print_matrix();
}
static void read_options(int argc,char** argv)
{
	char* temp;
	temp = *argv;
	while (++argv,--argc>0)
	{
		if (**argv == '-')
		{
			switch (*++*argv)
			{
			case 'o':
				--argc;
				option = atoi(*++argv);	// fast or rand
				break;
			case 'l':
				--argc;
				difflimit = atof(*++argv);	// difflimit
				break;
			case 'p':
				--argc;
				print_r = 1;
				break;
			case 'f':
				--argc;
				file_name = *++argv;
				break;
			case 'u':
				printf("\n Usage: [-o] 0 is fast,1 is rand\n");
				printf("	[-l] difference limit >0\n");
				printf("	[-p] print result matrix\n");
				exit(0);
				break;
			default:
				break;
			}
		}
	}
	
}
static void print_matrix(void)
{
	int i, j;

	for (i = 0; i < real_size+2; i++) {
		for (j = 0; j < real_size+2; j++)
			printf(" %7.2f", matrixA[i][j]);
		printf("\n");
	}
	printf("\n");
}