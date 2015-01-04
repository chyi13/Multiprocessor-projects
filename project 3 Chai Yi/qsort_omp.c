/*
	openmp quick sort
	Chai Yi 2014/5/20
*/
#include <omp.h>
#include <stdio.h>

#define MAX_ITEMS (64*(1024*1024)) /* 64 Mitems */
#define MAX_THREADS 32
#define swap(v, a, b) {int tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}

static int* v;
static int v_temp[MAX_ITEMS];
static int head_tail[MAX_THREADS*2];	// every thread's first and last elements indices
int PRINT = 0;
int np = 8; 	// processor number

static void init_array(void);
static int partition(int *v, int low, int high, int pivot_index);
int read_options(int argc, char **argv);
static void print_array(void);
static void division(int LOW,int HIGH);
static void quick_sort(int *v, int low, int high);
static int partition_seq(int *v, int low, int high, int pivot_index);
int main(int argc, char * argv[])
{
	omp_set_num_threads(8);			// fixed 8 threads
	read_options(argc,argv);
	init_array();
	if (PRINT)
		print_array();
		
	int i,j;
	int total_left = 0, total_right = 0;
	
	// 1111111111111111111111111111111111111111111111111111111111111111111111111
	int pivot_value = 1073741824;
	
	printf("p: %d\n",pivot_value);
	// local re-arrangement
	int left_num[MAX_THREADS];
	int right_num[MAX_THREADS];
	#pragma omp parallel shared(pivot_value,left_num)
	{
		int temp_np = omp_get_thread_num();
		left_num[temp_np] = partition(v,head_tail[temp_np*2],head_tail[temp_np*2+1],pivot_value) - head_tail[temp_np*2];
		right_num[temp_np] = head_tail[temp_np*2+1]-head_tail[temp_np*2]+1-left_num[temp_np];
	}

	for (i = 0; i<np; i++)
		total_left+= left_num[i];
	total_right = MAX_ITEMS - total_left;
	
	int prefix_left[MAX_THREADS] = {0};
	int prefix_right[MAX_THREADS] = {0};
	prefix_right[0] = total_left;
	for (i = 1; i<np; i++)				// prefix sum
	{
		prefix_left[i] += prefix_left[i-1] + left_num[i-1];
		prefix_right[i]+= prefix_right[i-1] + right_num[i-1];
	}
	
	// global re-arrangement
	memset(v_temp,0,MAX_ITEMS*sizeof(int));
	
	#pragma omp parallel for shared(v_temp)
	for (i = 0; i<np; i++)
	{	
		int it1 = head_tail[i*2],it2 = prefix_left[i];
		int temp_count = left_num[i];
		while(temp_count>0)
		{
			temp_count--;
			v_temp[it2] = v[it1];
			it1++;
			it2++;
		}
		temp_count = right_num[i];
		it1 = head_tail[i*2]+left_num[i];
		it2 = prefix_right[i];
		while(temp_count>0)
		{
			temp_count--;
			v_temp[it2] = v[it1];
			it1++;
			it2++;
		}
	}
	memcpy(v,v_temp,sizeof(int)*MAX_ITEMS);
	if (PRINT)
	{
		printf("1st step: ");
		print_array();
	}
	
	// 2222222222222222222222222222222222222222222222222222222222222222222222222222
	int pivot_value1 = pivot_value/2, pivot_value2 = pivot_value/2 + pivot_value;	// pivot value
	int items_per_thread1 = total_left/(np/2), items_per_thread2 = total_right/(np/2);

	for (i = 0,j = 0; i<np/2; i++, j+=items_per_thread1)
	{
		head_tail[i*2] = j;
		if (i != np/2-1)
			head_tail[i*2+1] = j+ items_per_thread1-1;
		else
			head_tail[i*2+1] = total_left - 1;			// in case for not divisible items
	}
	
	for (i = np/2, j = total_left; i<np; i++, j+= items_per_thread2)
	{
		head_tail[i*2] = j;
		if (i != np-1)
			head_tail[i*2+1] = j + items_per_thread2-1;
		else
			head_tail[i*2+1] = MAX_ITEMS-1;				// in case for not divisible items
	}
	
	// local re-arrangement
	#pragma omp parallel shared(pivot_value1,pivot_value2,left_num,right_num)
	{
		int temp_np = omp_get_thread_num();
		if (temp_np<np/2)
		{
			left_num[temp_np] = partition(v,head_tail[temp_np*2],head_tail[temp_np*2+1],pivot_value1) - head_tail[temp_np*2];
			right_num[temp_np] = head_tail[temp_np*2+1]-head_tail[temp_np*2]+1-left_num[temp_np];
		}
		else
		{
			left_num[temp_np] = partition(v,head_tail[temp_np*2],head_tail[temp_np*2+1],pivot_value2) - head_tail[temp_np*2];
			right_num[temp_np] = head_tail[temp_np*2+1]-head_tail[temp_np*2]+1-left_num[temp_np];
		}
	}
	
	int total_left1 = 0,total_left2 = 0,total_right1,total_right2;	// sub-section count: left1+right1 = left left2+right2=right
	for (i = 0; i<np/2; i++)
	{
		total_left1+= left_num[i];
	}
	for (i = np/2; i<np; i++)
	{
		total_left2+= left_num[i];
	}
	total_right1 = total_left - total_left1;
	total_right2 = total_right - total_left2;
	
	// global re-arrangement
	memset(v_temp,0,sizeof(int)*MAX_ITEMS);
	memset(prefix_left,0,sizeof(int)*MAX_THREADS);
	memset(prefix_right,0,sizeof(int)*MAX_THREADS);
	
	prefix_left[np/2] = total_left;
	prefix_right[0] = total_left1;
	prefix_right[np/2] = total_left + total_left2;

	for (i = 1; i<np/2; i++)		// prefix sum of left part 
	{
		prefix_left[i] += prefix_left[i-1] + left_num[i-1];
		prefix_right[i]+= prefix_right[i-1] + right_num[i-1];
	}
	for (i=np/2+1; i<np; i++)		// prefix sum of right part
	{
		prefix_left[i] += prefix_left[i-1] + left_num[i-1];
		prefix_right[i]+= prefix_right[i-1] + right_num[i-1];
	}
	
	#pragma omp parallel for shared(v_temp)
	for (i = 0; i<np; i++)
	{	
		int it1 = head_tail[i*2],it2 = prefix_left[i];
		int temp_count = left_num[i];
		while(temp_count>0)
		{
			temp_count--;
			v_temp[it2] = v[it1];
			it1++;
			it2++;
		}	
		temp_count = right_num[i];
		it1 = head_tail[i*2]+left_num[i];
		it2 = prefix_right[i];
		while(temp_count>0)
		{
			temp_count--;
			v_temp[it2] = v[it1];
			it1++;
			it2++;
		}	
	}
	memcpy(v,v_temp,sizeof(int)*MAX_ITEMS);

	if (PRINT)
	{	
		printf("2nd step: ");
		print_array();
	}
	
	// sequential process  4 threads
	#pragma omp parallel num_threads(4)
	{
		int temp_np = omp_get_thread_num();
		if (temp_np == 0)
			quick_sort(v,0,total_left1-1);
		if (temp_np == 1)
			quick_sort(v,total_left1, total_left-1);
		if (temp_np == 2)
			quick_sort(v,total_left, total_left+total_left2-1);
		if (temp_np == 3)
			quick_sort(v,total_left+total_left2,MAX_ITEMS-1);
	}
	
	if (PRINT)
		print_array();
}
static void init_array(void)
{	
	// initialize target vector
    int i,j;
    v = (int *) malloc(MAX_ITEMS*sizeof(int));
	int items_per_thread = MAX_ITEMS/np;
	
//	#pragma omp parallel for schedule(static,1024*1024*8)
	for (i = 0; i < MAX_ITEMS; i++)
		v[i] = rand();
	
	// initialize head & tail of each thread 
	for (i = 0,j = 0; i<MAX_ITEMS; i+=items_per_thread,j+=2)
	{
		head_tail[j] = i;
		head_tail[j+1] = i+items_per_thread-1;
	}
	if (PRINT)
		printf("Initialization completed\n");
}


static int partition(int *v, int low, int high, int pivot_value)
{
//	printf("ppp %d %d \n",low,high);

    /* move elements into place */
    while (low <= high) {
        if (v[low] <= pivot_value)
            low++;
        else if (v[high] > pivot_value)
            high--;
        else
		{
            swap(v, low, high);
		}
    }

    /* put pivot back between two groups */
  //  if (high != pivot_index)
  //     swap(v, pivot_index, high);
    return low;
}

int read_options(int argc, char **argv)
{
    char    *prog;
 
    prog = *argv;
    while (++argv, --argc > 0)
	if (**argv == '-')
	    switch ( *++*argv ) {
		case 'p':
			--argc;
			PRINT = atoi(*++argv);
			break;
		default:
		break;
		}
	return 1;
}
static void print_array(void)
{
    int i;

    for (i = 0; i < MAX_ITEMS; i++)
        printf("%d ", v[i]);
    printf("\n");
}

static int partition_seq(int *v, int low, int high, int pivot_index)
{
    /* move pivot to the bottom of the vector */
    if (pivot_index != low)
        swap(v, low, pivot_index);

    pivot_index = low;
    low++;

    /* invariant:
     * v[i] for i less than low are less than or equal to pivot
     * v[i] for i greater than high are greater than pivot
     */

    /* move elements into place */
    while (low <= high) {
        if (v[low] <= v[pivot_index])
            low++;
        else if (v[high] > v[pivot_index])
            high--;
        else
            swap(v, low, high);
    }

    /* put pivot back between two groups */
    if (high != pivot_index)
        swap(v, pivot_index, high);
    return high;
}

static void quick_sort(int *v, int low, int high)
{
    int pivot_index;
    
    /* no need to sort a vector of zero or one element */
    if (low >= high)
        return;

    /* select the pivot value */
    pivot_index = (low+high)/2;

    /* partition the vector */
    pivot_index = partition_seq(v, low, high, pivot_index);

    /* sort the two sub arrays */
    if (low < pivot_index)
        quick_sort(v, low, pivot_index-1);
    if (pivot_index < high)
        quick_sort(v, pivot_index+1, high);
}
