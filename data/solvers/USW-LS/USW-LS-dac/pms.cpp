/**
 * USW-LS-DAC: USW-LS MaxSAT solver with Dynamic Algorithm Configuration support.
 *
 * Based on USW-LS by Yi Chu et al.
 * Modified to support:
 *   1. DAC checkpoint protocol (--dac N): emit state every N flips, read params from stdin
 *   2. New WCNF 2022+ format (h prefix for hard clauses)
 *   3. Windows (MSVC) compilation
 *
 * Usage:
 *   Normal mode:  usw-ls-dac <instance.wcnf> <seed>
 *   DAC mode:     usw-ls-dac <instance.wcnf> <seed> --dac 10000
 *
 * DAC Protocol:
 *   1. Solver emits: DAC_READY <num_vars> <num_clauses> <num_hard> <num_soft> <total_soft_weight>
 *   2. At each checkpoint, solver emits:
 *      DAC_STATE <step> <hard_unsat> <soft_unsat_weight> <opt_unsat_weight> <total_step>
 *               <weight_mean> <weight_std> <feasible> <time>
 *               <hard_large_count> <soft_large_count> <goodvar_count>
 *   3. Controller responds with one of:
 *      - "CONTINUE"  (keep current parameters)
 *      - "STOP"      (terminate solver)
 *      - "<h_inc> <s_inc> <smooth_prob> <soft_smooth_prob> <rwprob> <rdprob>"
 */

#include "basis_pms.h"
#include "build.h"
#include "heuristic.h"
#include <signal.h>

USW s;
int seed = 1;
long long best_known;
long long total_step = 0;
long long consecutive_better_soft = 0;
char * file_name = NULL;

void interrupt(int sig)
{
    if (s.verify_sol() == 1)
        cout << "c verified" << endl;
    s.print_best_solution();
    s.free_memory();
    exit(10);
}

int main(int argc, char *argv[])
{
    start_timing();

    signal(SIGTERM, interrupt);
#ifndef _WIN32
    // SIGALRM not available on Windows
    signal(SIGALRM, interrupt);
#endif

    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <instance.wcnf> <seed> [--dac <interval>] [options...]" << endl;
        return 1;
    }

    sscanf(argv[2], "%d", &seed);
    srand(seed);
    s.build_instance(argv[1]);

    s.settings();

    s.parse_parameters2(argc, argv);
    s.local_search_with_decimation(argv[1]);

    // Print final result
    if (s.verify_sol() == 1)
        cout << "c verified" << endl;
    s.print_best_solution();

    return 0;
}
