/* Reference trace of ONA's Pong and Pong2 game loops.
 *
 * Compiles ONA's src/system_tests/Pong2_Test.h and Pong_Test.h unchanged,
 * with the NAR calls stubbed: inputs are printed as TRACE lines, and the
 * NAR_Cycles(5) at the end of each step stands in for the reasoner with a
 * scripted policy drawn from the game's own generator (myrand() % 4: left,
 * right, stop, nothing). onabench.pong replays the same policy, so both
 * print the same trace when the port's rules are ONA's.
 *
 *   cc -I $ONA/src/system_tests -o pong_trace pong_trace.c
 *   ./pong_trace pong|pong2 ITERATIONS SEED
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLEAR_SCREEN do{} while(0)
#define SLEEP do{} while(0)

static unsigned long next = 1;
int myrand(void) { next = next * 1103515245 + 12345; return((unsigned)(next/65536) % 32768); }
void mysrand(unsigned int seed) { next = seed; }

typedef struct { int unused; } Feedback;
typedef const char *Term;
static Term Narsese_AtomicTerm(const char *s) { return s; }
static void NAR_AddInputBelief(Term t) { printf("TRACE belief %s\n", t); }
static void NAR_AddInputGoal(Term t) { printf("TRACE goal %s\n", t); }
static void NAR_AddOperation(const char *name, Feedback (*f)(void)) { (void) name; (void) f; }
static void NAR_Cycles(int n);

#include "Pong2_Test.h"
#include "Pong_Test.h"

static void NAR_Cycles(int n)
{
    (void) n;
    int r = myrand() % 4;
    printf("TRACE policy %d\n", r);
    if(r == 0) NAR_Pong_Left();
    if(r == 1) NAR_Pong_Right();
    if(r == 2) NAR_Pong_Stop();
}

int main(int argc, char *argv[])
{
    long iterations = atol(argv[2]);
    mysrand((unsigned) atol(argv[3]));
    if(!strcmp(argv[1], "pong")) NAR_Pong(iterations);
    else NAR_Pong2(iterations);
    return 0;
}
