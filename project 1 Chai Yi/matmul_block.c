// author:Chai Yi chyi13@student.bth.se
// date: 2014/04/25
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

#define SIZE 1024	/* assumption: SIZE a multiple of number of nodes */
/* Hint: use small sizes when testing, e.g., SIZE 8 */
#define FROM_MASTER 1	/* setting a message type */
#define FROM_WORKER 2	/* setting a message type */
#define DEBUG 0		/* 1 = debug on, 0 = debug off */

MPI_Status status;

static double a[SIZE][SIZE];
static double b[SIZE][SIZE];
static double c[SIZE][SIZE];
static double d[SIZE][SIZE];	// temp storage


static void init_matrix(void)
{
	int i, j;
	for (i = 0; i < SIZE; i++)
		for (j = 0; j < SIZE; j++) {
			/* Simple initialization, which enables us to easily check
			* the correct answer. Each element in c will have the same 
			* value as SIZE after the matmul operation.
			*/
			a[i][j] = 1;
			b[i][j] = 1;
		}
}

static void print_matrix(void)
{
	int i, j;

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++)
			printf(" %7.2f", c[i][j]);
		printf("\n");
	}
}

static void calcu_rows_cols(int mnproc, int* rows,int* cols)	
{
	int temprows = sqrt((float)mnproc);
	while(mnproc%temprows!=0)
		temprows--;

	*rows = temprows;
	*cols = mnproc/temprows;
}
static void transpose(double matrix[SIZE][SIZE])
{
	int i,j;
	for (i = 0; i<SIZE; i++)
	{
		for (j = 0; j<SIZE; j++)
		{
			d[j][i] = matrix[i][j];
		}
	}
	memcpy(matrix,d,SIZE*SIZE*sizeof(double));
}
int main(int argc, char **argv)
{
	int myrank, nproc;
	int rows;	/* amount of work per node (rows per worker) */
	int cols;	/* amount of work per node (cols per worker) */
	int mtype; /* message type: send/recv between master and workers */
	int dest, src;
	int col_size, row_size;

	double start_time, end_time;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	if (myrank == 0) // master node
	{
		/* Initialization */
		init_matrix();
		start_time = MPI_Wtime();

		/* Send block part of matrix a and the whole matrix b to workers */
		calcu_rows_cols(nproc, &rows, &cols);

		/////////////////////////////////////////
		// transpose
		transpose(b);
		int i,j;
		if (DEBUG)
		{
			printf("Transpose matrix B:\n");
			for (i = 0; i < SIZE; i++) {
				for (j = 0; j < SIZE; j++)
					printf(" %7.2f", b[i][j]);
				printf("\n");
			}
			printf("small rows: %d, cols: %d\n",rows,cols);
		}
		row_size = SIZE/rows;
		col_size = SIZE/cols;
		if (DEBUG)
			printf("rowsize: %d, colsize: %d\n",row_size,col_size);
		mtype = FROM_MASTER;

		int temprow,tempcol;
		for (dest = 1; dest<nproc; dest++)
		{
			temprow = dest/cols;
			tempcol = dest-temprow*cols;
			temprow = temprow*row_size;
			tempcol = tempcol*col_size;

			if (DEBUG)
				printf("n:%d row: %d, col: %d\n",dest,temprow,tempcol);

			MPI_Send(&temprow,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&tempcol,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&row_size,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&col_size,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&a[temprow][0],row_size*SIZE,MPI_DOUBLE,dest,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[tempcol][0],col_size*SIZE,MPI_DOUBLE,dest,mtype,MPI_COMM_WORLD);
		}

		/* let master do its part of the work */
		for (i= 0; i<row_size; i++)
		{
			for (j= 0; j<col_size; j++)
			{
				c[i][j] = 0;
				int k;
				for (k= 0; k<SIZE; k++)
				{
					c[i][j]+=a[i][k]*b[j][k];
				}
			}
		}

		/* collect the results from all the workers */
		mtype = FROM_WORKER;
		for (src = 1; src < nproc; src++) {
			MPI_Recv(&temprow, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&tempcol, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);

			MPI_Recv(&d[temprow][0], row_size*SIZE, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);

			for (i = temprow; i<temprow+row_size; i++)
			{
				for (j = tempcol; j<tempcol+col_size; j++)
					c[i][j] = d[i][j];
			}
		}

		end_time = MPI_Wtime();
		printf("Execution time on %2d nodes: %f\n", nproc, end_time-start_time);
	
		if (DEBUG)
			print_matrix();
	}
	else
	{
		/* Worker tasks */
		/* Receive data from master */
		int temprow,tempcol;
		int row_size,col_size;
		mtype = FROM_MASTER;
		MPI_Recv(&temprow, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&tempcol, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&row_size, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&col_size, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&a[temprow][0], row_size*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&b[tempcol][0], col_size*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);

		/* do the workers part of the calculation */
		int i,j;
		for (i= temprow; i<temprow+row_size; i++)
		{
			for (j= tempcol; j<tempcol+col_size; j++)
			{
				c[i][j] = 0;
				int k;
				for (k= 0; k<SIZE; k++)
					c[i][j]+=a[i][k]*b[j][k];
			}
		}
		/* send the results to the master */
		mtype = FROM_WORKER;
		MPI_Send(&temprow, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
		MPI_Send(&tempcol, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
		MPI_Send(&c[temprow][0], row_size*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
	}
	MPI_Finalize();

	return 0;
}