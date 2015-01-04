/*****************************************************
*
* Gaussian elimination
*
* sequential version
*
****************************************************
*	Parallel version(OpenMP) implemented by
*     Chai Yi 
*     Date: 2014-5-25
*
****************************************************/

#include <stdio.h>
#include <omp.h>
#define MAX_SIZE 4096

typedef double matrix[MAX_SIZE][MAX_SIZE];
#define MAX_THREAD_NUM 32

int	N;		/* matrix size		*/
int	maxnum;		/* max number of element*/
char	*Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */


int thread_num = 8;

/* forward declarations */
void work(void);

void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char **);

int main(int argc, char **argv)
{
	int i, timestart, timeend, iter;

	Init_Default();		/* Init default values	*/
	Read_Options(argc,argv);	/* Read arguments	*/
	Init_Matrix();		/* Init the matrix	*/
	work();
	if (PRINT == 1)
		Print_Matrix();
	return 0;
}

void work()
{
	#pragma omp parallel
	{
		int k;
	//	#pragma omp for
		for (k = 0; k<N; k++)
		{
			int i,j;
			int temp_np = omp_get_thread_num();
		
			// first 
			if ((k%thread_num) == temp_np)
			{
				for (j = k+1; j<N; j++)
					A[k][j] = A[k][j]/A[k][k];
				y[k] = b[k] / A[k][k];
				A[k][k] = 1;
			}	
		
			#pragma omp barrier
			
			// Second
			for (i = k+1; i<N; i++)
			{
				if ((i%thread_num) == temp_np)
				{
					for (j = k+1; j<N; j++)
					{
						A[i][j] = A[i][j]- A[i][k]*A[k][j];
					}
					b[i] = b[i] - A[i][k]*y[k];
					A[i][k] = 0;
				}
			}
			#pragma omp barrier
		}
	}
}
void Init_Matrix()
{
	int i, j;
	printf("\nthread_num = %d",thread_num);
	printf("\nsize      = %dx%d ", N, N);
	printf("\nmaxnum    = %d \n", maxnum);
	printf("Init	  = %s \n", Init);
	printf("Initializing matrix...");

	if (strcmp(Init,"rand") == 0) {
		for (i = 0; i < N; i++){
			for (j = 0; j < N; j++) {
				if (i == j) /* diagonal dominance */
					A[i][j] = (double)(rand() % maxnum) + 5.0;
				else
					A[i][j] = (double)(rand() % maxnum) + 1.0;
			}
		}
	}
	if (strcmp(Init,"fast") == 0) {
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				if (i == j) /* diagonal dominance */
					A[i][j] = 5.0;
				else
					A[i][j] = 2.0;
			}
		}
	}

	/* Initialize vectors b and y */
	for (i = 0; i < N; i++) {
		b[i] = 2.0;
		y[i] = 1.0;
	}

	printf("done \n\n");
	if (PRINT == 1)
		Print_Matrix();
}

void Print_Matrix()
{
	int i, j;

	printf("Matrix A:\n");
	for (i = 0; i < N; i++) {
		printf("[");
		for (j = 0; j < N; j++)
			printf(" %5.2f,", A[i][j]);
		printf("]\n");
	}
	printf("Vector b:\n[");
	for (j = 0; j < N; j++)
		printf(" %5.2f,", b[j]);
	printf("]\n");
	printf("Vector y:\n[");
	for (j = 0; j < N; j++)
		printf(" %5.2f,", y[j]);
	printf("]\n");
	printf("\n\n");
}

void Init_Default()
{
	N = 8;
	Init = "rand";
	maxnum = 15.0;
	PRINT = 1;
}

int Read_Options(int argc, char **argv)
{
	char    *prog;

	prog = *argv;
	while (++argv, --argc > 0)
		if (**argv == '-')
			switch ( *++*argv ) {
			case 'p':
				--argc;
				thread_num =  atoi(*++argv);		// thread number
				break;
			case 'n':
				--argc;
				N = atoi(*++argv);
				break;
			case 'h':
				printf("\nHELP: try sor -u \n\n");
				exit(0);
				break;
			case 'u':
				printf("\nUsage: sor [-n problemsize]\n");
				printf("           [-D] show default values \n");
				printf("           [-h] help \n");
				printf("           [-I init_type] fast/rand \n");
				printf("           [-m maxnum] max random no \n");
				printf("           [-P print_switch] 0/1 \n");
				exit(0);
				break;
			case 'D':
				printf("\nDefault:  n         = %d ", N);
				printf("\n          Init      = rand" );
				printf("\n          maxnum    = 5 ");
				printf("\n          P         = 0 \n\n");
				exit(0);
				break;
			case 'I':
				--argc;
				Init = *++argv;
				break;
			case 'm':
				--argc;
				maxnum = atoi(*++argv);
				break;
			case 'P':
				--argc;
				PRINT = atoi(*++argv);
				break;
			default:
				printf("%s: ignored option: -%s\n", prog, *argv);
				printf("HELP: try %s -u \n\n", prog);
				break;
		} 
	return 1;
}