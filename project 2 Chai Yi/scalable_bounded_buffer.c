/*****************************************************************************
*
* bounded_buffer_simple.c
*
* Implementation of a producer-consumer scenario using a bounded buffer
*
* Problem to be solved:
*    The buffer management has (at least) one severe performance bottleneck.
*    Your task is to solve the bottleneck(s).
*    The resulting code shall have a speedup almost as high as the 
*    number of cpus.
*
****************************************************************************
*
*	revised by Chai Yi chyi13@student.bth.se
*	date: 2014/05/10
****************************************************************************/


#include <stdio.h>
#include <pthread.h>
#include <errno.h>
#define MAX_THREAD_NUM 32			// maximum thread number
#define SINGLE_BUFFER_SIZE 10		// single buffer size is 10
#define KILO 1024
#define MEGA (KILO*KILO)
#define ITEMS_TO_SEND 8*MEGA /* number of items to pass through the buffer */
//#define ITEMS_TO_SEND 128

static int print_flag = 0; /* 1 = printouts, 0 = no printouts */

int thread_num;				// user determined running threads number
int items_per_thread;		// items per thread to be consumed or produced

typedef struct {
	int buf[MAX_THREAD_NUM][SINGLE_BUFFER_SIZE];	// total buffer = thread_num * SINGLE_BUFFER_SIZE
	int in[MAX_THREAD_NUM];			// producer item index
	int out[MAX_THREAD_NUM];		// consumer item index
	int no_elems[MAX_THREAD_NUM];	// current item number in the buffer
	int no_items_sent[MAX_THREAD_NUM];	// items already sent per producer thread
	int no_items_received[MAX_THREAD_NUM];	// items already received per consumer thread
	
	pthread_mutex_t lock[MAX_THREAD_NUM]; 	// protects the buffer
	pthread_cond_t cond_full[MAX_THREAD_NUM];	// condition full to inform consumer that the buffer is not empty
} buffer_t;

static buffer_t buffer;

void init_buffer(void)
{
	items_per_thread = ITEMS_TO_SEND/thread_num;		// items per thread
	
	int i = 0;
	for (i =0; i<thread_num; i++)
	{
		buffer.no_elems[i] = 0;
		buffer.in[i] = 0;
		buffer.out[i] = 0;
		buffer.no_items_sent[i] = 0;
		buffer.no_items_received[i] = 0;
		
		pthread_mutex_init(&buffer.lock[i], NULL);	// initialize all mutex
		pthread_cond_init(&buffer.cond_full[i],NULL);	// initialize condition
	}
}

void *consumer(void *thr_id)
{
	int item;
	long my_id = (long) thr_id;

	//  if (print_flag)
	//    printf("Cstart 4: tid %d\n", my_id);
	while(1) {
		pthread_mutex_lock(&buffer.lock[my_id]);
		/* check if there is empty buffer places */
		if (buffer.no_elems[my_id]==0){
			pthread_cond_wait(&buffer.cond_full[my_id],&buffer.lock[my_id]);	// if no items in the buffer, then wait
		}
		
		//delay_in_buffer();
		item = buffer.buf[my_id][buffer.out[my_id]];						// consume
		buffer.out[my_id] = (buffer.out[my_id]+1)%SINGLE_BUFFER_SIZE;
		buffer.no_elems[my_id]--;
		buffer.no_items_received[my_id]++;
		
		if (print_flag){	
			printf("Consumer %d got number %d from buffer\n",my_id, item);
			fflush(stdout);
		}
		
		pthread_mutex_unlock(&buffer.lock[my_id]);
		
		if (buffer.no_items_received[my_id] == items_per_thread){		// if finish its work, exit
			break;
		}
		
	}
	if (print_flag)
		printf("CBreak 4: tid %d\n", my_id);
	pthread_exit(0);
}

void *producer(void *thr_id)
{
	int item;
	long my_id = (long) thr_id;

	if (print_flag)
	   printf("Pstart 4: tid %d\n", my_id);
	while(1) {
		pthread_mutex_lock(&buffer.lock[my_id]);
		
		/* check if there is empty buffer places */
		if (buffer.no_elems[my_id]<SINGLE_BUFFER_SIZE){
			item = items_per_thread*my_id + buffer.no_items_sent[my_id]; // calculate item index
			buffer.no_items_sent[my_id]++;
			
			buffer.buf[my_id][buffer.in[my_id]] = item;
			buffer.in[my_id] = (buffer.in[my_id] + 1) % SINGLE_BUFFER_SIZE;
			buffer.no_elems[my_id]++;
			
			pthread_cond_signal(&buffer.cond_full[my_id]);			// inform consumer 
			
			if (print_flag){
				printf("Producer %d put number %d in buffer\n", my_id, item);
				fflush(stdout);
			}
		}

		pthread_mutex_unlock(&buffer.lock[my_id]);
		
		if (buffer.no_items_sent[my_id] == items_per_thread)	// exit
			break;
		
	}
	
	if (print_flag)
		printf("PBreak 4: tid %d\n", my_id);
	pthread_exit(0);
}
void read_options(int argc, char **argv)
{
	char    *prog;
	while (++argv, --argc > 0)
	{
		if (**argv == '-')
			switch ( *++*argv ) {
			case 'n':
				--argc;
				thread_num = atoi(*++argv);		// number of thread
				break;
			case 'p':
				--argc;
				print_flag = atoi(*++argv);		// print flag
				break;
			default:
				break;
			}
	}
	
}
int main(int argc, char **argv)
{
	long i;
	pthread_t prod_thrs[MAX_THREAD_NUM];
	pthread_t cons_thrs[MAX_THREAD_NUM];
	pthread_attr_t attr;
	
	read_options(argc,argv);	// read options
	init_buffer();
	pthread_attr_init (&attr);

	printf("Buffer size = %d, items to send = %d\n", 
		thread_num, ITEMS_TO_SEND);
	/* create the producer and consumer threads */
	for(i = 0; i < thread_num; i++)
		pthread_create(&prod_thrs[i], &attr, producer, (void *)i);
	for(i = 0; i < thread_num; i++)
		pthread_create(&cons_thrs[i], &attr, consumer, (void *)i);

	/* wait for all threads to terminate */
	for (i = 0; i < thread_num; i++)
		pthread_join(prod_thrs[i], NULL);
	for (i = 0; i < thread_num; i++)
		pthread_join(cons_thrs[i], NULL);
	return 0;
}