#ifndef _BUILD_H_
#define _BUILD_H_

#include "basis_pms.h"
#include "pms.h"
#include <limits.h>
#include <vector>
#include <string>

extern int seed;
extern long long best_known;
extern long long total_step;
extern long long consecutive_better_soft;
extern char * file_name;

ISDist::ISDist() {
    dac_mode = 0;
    dac_checkpoint_interval = 0;
}

bool ISDist::parse_parameters2(int argc, char **argv)
{
    int i = 0;
    for (i = 1; i < argc; i++)
    {
        if (0 == strcmp(argv[i], "-best"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%lld", &best_known);
            cout << "c best_known: " << best_known << endl;
            if (best_known == -1)
            {
                cout << "c no feasible solution" << endl;
                exit(0);
            }
        }
        else if (0 == strcmp(argv[i], "-rdprob"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%f", &rdprob);
        }
        else if (0 == strcmp(argv[i], "-bms_num"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%d", &hd_count_threshold);
        }
        else if (0 == strcmp(argv[i], "-rwprob"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%f", &rwprob);
        }
        else if (0 == strcmp(argv[i], "-hard_sp"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%f", &smooth_probability);
        }
        else if (0 == strcmp(argv[i], "-soft_sp"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%f", &soft_smooth_probability);
        }
        else if (0 == strcmp(argv[i], "-soft_weight_threshold"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%lf", &softclause_weight_threshold);
        }
        else if (0 == strcmp(argv[i], "-h_inc"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%lf", &h_inc);
        }
        else if (0 == strcmp(argv[i], "-s_inc"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%lf", &s_inc);
        }
        else if (0 == strcmp(argv[i], "-coe"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%d", &coe_soft_clause_weight);
        }
        else if (0 == strcmp(argv[i], "--dac"))
        {
            i++;
            if (i >= argc) return false;
            sscanf(argv[i], "%d", &dac_checkpoint_interval);
            if (dac_checkpoint_interval > 0)
                dac_mode = 1;
            cout << "c DAC mode enabled, checkpoint every " << dac_checkpoint_interval << " flips" << endl;
        }
    }
    return true;
}

void ISDist::settings()
{
    best_known = -2;
    local_soln_feasible = 1;
    cutoff_time = 300;
    max_tries = 100000000;
    max_flips = 200000000;
    max_non_improve_flip = 10000000;
    large_clause_count_threshold = 0;
    soft_large_clause_count_threshold = 0;

    if (1 == problem_weighted) // Weighted Partial MaxSAT
    {
        coe_soft_clause_weight = 3000;
        if (0 != num_hclauses)
        {
            hd_count_threshold = 25;
            rdprob = 0.036;
            rwprob = 0.48;
            soft_smooth_probability = 9.9E-7;
            smooth_probability = 6.8E-5;
            h_inc = 30;
            s_inc = 10;
            softclause_weight_threshold = 200;
        }
        else
        {
            softclause_weight_threshold = 0;
            soft_smooth_probability = 1E-3;
            hd_count_threshold = 22;
            rdprob = 0.036;
            rwprob = 0.48;
            s_inc = 1.0;
        }
    }
    else // Unweighted Partial Maxsat
    {
        if (0 != num_hclauses)
        {
            hd_count_threshold = 96;
            coe_soft_clause_weight = 1000;
            h_inc = 1;
            s_inc = 1;
            rdprob = 0.079;
            rwprob = 0.087;
            soft_smooth_probability = 6.6E-5;
            softclause_weight_threshold = 183;
            smooth_probability = 2E-4;
        }
        else
        {
            hd_count_threshold = 94;
            coe_soft_clause_weight = 397;
            s_inc = 1;
            rdprob = 0.007;
            rwprob = 0.047;
            soft_smooth_probability = 0.002;
            softclause_weight_threshold = 550;
        }
    }
}

void ISDist::build_neighbor_relation()
{
    int i, j, count;
    int v, c, n;
    int temp_neighbor_count;

    for (v = 1; v <= num_vars; ++v)
    {
        neighbor_flag[v] = 1;
        temp_neighbor_count = 0;

        for (i = 0; i < var_lit_count[v]; ++i)
        {
            c = var_lit[v][i].clause_num;
            for (j = 0; j < clause_lit_count[c]; ++j)
            {
                n = clause_lit[c][j].var_num;
                if (neighbor_flag[n] != 1)
                {
                    neighbor_flag[n] = 1;
                    temp_neighbor[temp_neighbor_count++] = n;
                }
            }
        }

        neighbor_flag[v] = 0;

        var_neighbor[v] = new int[temp_neighbor_count];
        var_neighbor_count[v] = temp_neighbor_count;

        count = 0;
        for (i = 0; i < temp_neighbor_count; i++)
        {
            var_neighbor[v][count++] = temp_neighbor[i];
            neighbor_flag[temp_neighbor[i]] = 0;
        }
    }
}

void ISDist::build_instance(char *filename)
{
    total_soft_length = 0;
    total_hard_length = 0;
    total_soft_weight = 0;
    istringstream iss;
    string line;
    char tempstr1[10];
    char tempstr2[10];

    ifstream infile(filename);
    if (!infile)
    {
        cout << "c the input filename " << filename << " is invalid, please input the correct filename." << endl;
        exit(-1);
    }

    // Detect format: peek at first non-comment line
    bool old_format = false;
    bool format_detected = false;

    // First pass: detect format and count clauses for new format
    vector<string> all_lines;
    int new_fmt_num_vars = 0;
    int new_fmt_num_clauses = 0;

    while (getline(infile, line))
    {
        if (line.empty() || line[0] == 'c')
            continue;
        if (line[0] == 'p')
        {
            old_format = true;
            format_detected = true;
            // Parse old format header
            int read_items;
            num_vars = num_clauses = 0;
            read_items = sscanf(line.c_str(), "%s %s %d %d %lld", tempstr1, tempstr2, &num_vars, &num_clauses, &top_clause_weight);
            if (read_items < 5)
            {
                cout << "read item < 5 " << endl;
                exit(-1);
            }
            break;
        }
        // New format: no 'p' line, starts with 'h' or weight
        if (!format_detected)
        {
            format_detected = true;
            // Store this line and continue collecting
            all_lines.push_back(line);
            // Count vars from this line
            if (line[0] == 'h')
            {
                istringstream tmp(line.substr(1));
                int val;
                while (tmp >> val && val != 0)
                {
                    if (abs(val) > new_fmt_num_vars)
                        new_fmt_num_vars = abs(val);
                }
            }
            else
            {
                istringstream tmp(line);
                long long w;
                tmp >> w;
                int val;
                while (tmp >> val && val != 0)
                {
                    if (abs(val) > new_fmt_num_vars)
                        new_fmt_num_vars = abs(val);
                }
            }
            new_fmt_num_clauses++;
            continue;
        }
        // Continue collecting for new format
        if (!old_format)
        {
            all_lines.push_back(line);
            if (line[0] == 'h')
            {
                istringstream tmp(line.substr(1));
                int val;
                while (tmp >> val && val != 0)
                {
                    if (abs(val) > new_fmt_num_vars)
                        new_fmt_num_vars = abs(val);
                }
            }
            else
            {
                istringstream tmp(line);
                long long w;
                tmp >> w;
                int val;
                while (tmp >> val && val != 0)
                {
                    if (abs(val) > new_fmt_num_vars)
                        new_fmt_num_vars = abs(val);
                }
            }
            new_fmt_num_clauses++;
        }
    }

    if (!old_format)
    {
        // New format: set dimensions
        num_vars = new_fmt_num_vars;
        num_clauses = new_fmt_num_clauses;
        // Use a large sentinel for top_clause_weight
        top_clause_weight = (long long)1e18;
        cout << "c New WCNF format detected: " << num_vars << " vars, " << num_clauses << " clauses" << endl;
    }

    allocate_memory();

    int v, c_idx;
    for (c_idx = 0; c_idx < num_clauses; c_idx++)
    {
        clause_lit_count[c_idx] = 0;
        clause_lit[c_idx] = NULL;
    }
    for (v = 1; v <= num_vars; ++v)
    {
        var_lit_count[v] = 0;
        var_lit[v] = NULL;
        var_neighbor[v] = NULL;
    }

    int cur_lit;
    c_idx = 0;
    problem_weighted = 0;
    num_hclauses = num_sclauses = 0;
    unit_clause_count = 0;

    int *redunt_test = new int[num_vars + 10];
    memset(redunt_test, 0, sizeof(int) * (num_vars + 10));

    if (old_format)
    {
        // Old format: read clauses from file stream (positioned after 'p' line)
        while (getline(infile, line))
        {
            if (line.empty() || line[0] == 'c')
                continue;

            iss.clear();
            iss.str(line);
            iss.seekg(0, ios::beg);
            clause_lit_count[c_idx] = 0;

            iss >> org_clause_weight[c_idx];
            if (org_clause_weight[c_idx] != top_clause_weight)
            {
                if (org_clause_weight[c_idx] != 1)
                    problem_weighted = 1;
                total_soft_weight += org_clause_weight[c_idx];
                num_sclauses++;
            }
            else
            {
                num_hclauses++;
            }

            iss >> cur_lit;
            int clause_reduent = 0;
            while (cur_lit != 0)
            {
                if (redunt_test[abs(cur_lit)] == 0)
                {
                    temp_lit[clause_lit_count[c_idx]] = cur_lit;
                    clause_lit_count[c_idx]++;
                    redunt_test[abs(cur_lit)] = cur_lit;
                }
                else if (redunt_test[abs(cur_lit)] != cur_lit)
                {
                    clause_reduent = 1;
                    break;
                }
                iss >> cur_lit;
            }
            if (clause_reduent == 1)
            {
                for (int i = 0; i < clause_lit_count[c_idx]; ++i)
                    redunt_test[abs(temp_lit[i])] = 0;
                num_clauses--;
                clause_lit_count[c_idx] = 0;
                continue;
            }

            clause_lit[c_idx] = new lit[clause_lit_count[c_idx] + 1];
            int i;
            for (i = 0; i < clause_lit_count[c_idx]; ++i)
            {
                clause_lit[c_idx][i].clause_num = c_idx;
                clause_lit[c_idx][i].var_num = abs(temp_lit[i]);
                redunt_test[abs(temp_lit[i])] = 0;
                if (temp_lit[i] > 0)
                    clause_lit[c_idx][i].sense = 1;
                else
                    clause_lit[c_idx][i].sense = 0;
                var_lit_count[clause_lit[c_idx][i].var_num]++;
            }
            clause_lit[c_idx][i].var_num = 0;
            clause_lit[c_idx][i].clause_num = -1;

            if (clause_lit_count[c_idx] == 1)
                unit_clause[unit_clause_count++] = clause_lit[c_idx][0];
            if (top_clause_weight == org_clause_weight[c_idx])
                total_hard_length += clause_lit_count[c_idx];
            else
                total_soft_length += clause_lit_count[c_idx];
            c_idx++;
        }
    }
    else
    {
        // New format: process collected lines
        for (size_t li = 0; li < all_lines.size(); li++)
        {
            line = all_lines[li];
            if (line.empty() || line[0] == 'c')
                continue;

            clause_lit_count[c_idx] = 0;
            bool is_hard = false;

            if (line[0] == 'h')
            {
                // Hard clause
                org_clause_weight[c_idx] = top_clause_weight;
                num_hclauses++;
                is_hard = true;
                iss.clear();
                iss.str(line.substr(1)); // skip 'h'
                iss.seekg(0, ios::beg);
            }
            else
            {
                // Soft clause: "weight lit1 lit2 ... 0"
                iss.clear();
                iss.str(line);
                iss.seekg(0, ios::beg);
                iss >> org_clause_weight[c_idx];
                if (org_clause_weight[c_idx] != 1)
                    problem_weighted = 1;
                total_soft_weight += org_clause_weight[c_idx];
                num_sclauses++;
            }

            iss >> cur_lit;
            int clause_reduent = 0;
            while (cur_lit != 0)
            {
                if (redunt_test[abs(cur_lit)] == 0)
                {
                    temp_lit[clause_lit_count[c_idx]] = cur_lit;
                    clause_lit_count[c_idx]++;
                    redunt_test[abs(cur_lit)] = cur_lit;
                }
                else if (redunt_test[abs(cur_lit)] != cur_lit)
                {
                    clause_reduent = 1;
                    break;
                }
                iss >> cur_lit;
            }
            if (clause_reduent == 1)
            {
                for (int i = 0; i < clause_lit_count[c_idx]; ++i)
                    redunt_test[abs(temp_lit[i])] = 0;
                num_clauses--;
                clause_lit_count[c_idx] = 0;
                continue;
            }

            clause_lit[c_idx] = new lit[clause_lit_count[c_idx] + 1];
            int i;
            for (i = 0; i < clause_lit_count[c_idx]; ++i)
            {
                clause_lit[c_idx][i].clause_num = c_idx;
                clause_lit[c_idx][i].var_num = abs(temp_lit[i]);
                redunt_test[abs(temp_lit[i])] = 0;
                if (temp_lit[i] > 0)
                    clause_lit[c_idx][i].sense = 1;
                else
                    clause_lit[c_idx][i].sense = 0;
                var_lit_count[clause_lit[c_idx][i].var_num]++;
            }
            clause_lit[c_idx][i].var_num = 0;
            clause_lit[c_idx][i].clause_num = -1;

            if (clause_lit_count[c_idx] == 1)
                unit_clause[unit_clause_count++] = clause_lit[c_idx][0];
            if (is_hard)
                total_hard_length += clause_lit_count[c_idx];
            else
                total_soft_length += clause_lit_count[c_idx];
            c_idx++;
        }
    }

    delete[] redunt_test;
    infile.close();

    // Adjust num_clauses to actual count
    num_clauses = c_idx;

    // creat var literal arrays
    for (v = 1; v <= num_vars; ++v)
    {
        var_lit[v] = new lit[var_lit_count[v] + 1];
        var_lit_count[v] = 0;
    }
    for (int c = 0; c < num_clauses; ++c)
    {
        for (int i = 0; i < clause_lit_count[c]; ++i)
        {
            v = clause_lit[c][i].var_num;
            var_lit[v][var_lit_count[v]] = clause_lit[c][i];
            ++var_lit_count[v];
        }
    }
    for (v = 1; v <= num_vars; ++v)
        var_lit[v][var_lit_count[v]].clause_num = -1;

    build_neighbor_relation();

    best_soln_feasible = 0;

    cout << "c Instance: " << num_vars << " vars, " << num_clauses << " clauses ("
         << num_hclauses << " hard, " << num_sclauses << " soft)" << endl;
}

void ISDist::allocate_memory()
{
    int malloc_var_length = num_vars + 10;
    int malloc_clause_length = num_clauses + 10;

    unit_clause = new lit[malloc_clause_length];

    var_lit = new lit *[malloc_var_length];
    var_lit_count = new int[malloc_var_length];
    clause_lit = new lit *[malloc_clause_length];
    clause_lit_count = new int[malloc_clause_length];

    score = new double[malloc_var_length];
    var_neighbor = new int *[malloc_var_length];
    var_neighbor_count = new int[malloc_var_length];
    time_stamp = new long long[malloc_var_length];
    neighbor_flag = new int[malloc_var_length];
    temp_neighbor = new int[malloc_var_length];

    org_clause_weight = new long long[malloc_clause_length];
    clause_weight = new double[malloc_clause_length];
    tuned_org_clause_weight = new double[malloc_clause_length];
    sat_count = new int[malloc_clause_length];
    sat_var = new int[malloc_clause_length];
    clause_selected_count = new long long[malloc_clause_length];
    best_soft_clause = new int[malloc_clause_length];

    hardunsat_stack = new int[malloc_clause_length];
    index_in_hardunsat_stack = new int[malloc_clause_length];
    softunsat_stack = new int[malloc_clause_length];
    index_in_softunsat_stack = new int[malloc_clause_length];

    unsatvar_stack = new int[malloc_var_length];
    index_in_unsatvar_stack = new int[malloc_var_length];
    unsat_app_count = new int[malloc_var_length];

    goodvar_stack = new int[malloc_var_length];
    already_in_goodvar_stack = new int[malloc_var_length];

    cur_soln = new int[malloc_var_length];
    best_soln = new int[malloc_var_length];
    local_opt_soln = new int[malloc_var_length];

    large_weight_clauses = new int[malloc_clause_length];
    soft_large_weight_clauses = new int[malloc_clause_length];
    already_in_soft_large_weight_stack = new int[malloc_clause_length];

    best_array = new int[malloc_var_length];
    temp_lit = new int[malloc_var_length];
}

void ISDist::free_memory()
{
    int i;
    for (i = 0; i < num_clauses; i++)
        delete[] clause_lit[i];

    for (i = 1; i <= num_vars; ++i)
    {
        delete[] var_lit[i];
        delete[] var_neighbor[i];
    }

    delete[] var_lit;
    delete[] var_lit_count;
    delete[] clause_lit;
    delete[] clause_lit_count;

    delete[] score;
    delete[] var_neighbor;
    delete[] var_neighbor_count;
    delete[] time_stamp;
    delete[] neighbor_flag;
    delete[] temp_neighbor;

    delete[] org_clause_weight;
    delete[] clause_weight;
    delete[] tuned_org_clause_weight;
    delete[] sat_count;
    delete[] sat_var;
    delete[] clause_selected_count;
    delete[] best_soft_clause;

    delete[] hardunsat_stack;
    delete[] index_in_hardunsat_stack;
    delete[] softunsat_stack;
    delete[] index_in_softunsat_stack;

    delete[] unsatvar_stack;
    delete[] index_in_unsatvar_stack;
    delete[] unsat_app_count;

    delete[] goodvar_stack;
    delete[] already_in_goodvar_stack;

    delete[] cur_soln;
    delete[] best_soln;
    delete[] local_opt_soln;

    delete[] large_weight_clauses;
    delete[] soft_large_weight_clauses;
    delete[] already_in_soft_large_weight_stack;

    delete[] best_array;
    delete[] temp_lit;
}

#endif
