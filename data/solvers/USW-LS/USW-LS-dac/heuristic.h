#ifndef _HEURISTIC_H_
#define _HEURISTIC_H_

#include "basis_pms.h"
#include "deci.h"

extern int seed;
extern long long best_known;
extern long long total_step;
extern long long consecutive_better_soft;
extern char * file_name;

void USW::init(vector<int> &init_solution)
{
    soft_large_weight_clauses_count = 0;
    if (1 == problem_weighted) // weighted
    {
        if (0 != num_hclauses) // weighted partial
        {
            for (int c = 0; c < num_clauses; c++)
            {
                already_in_soft_large_weight_stack[c] = 0;
                clause_selected_count[c] = 0;

                if (org_clause_weight[c] == top_clause_weight)
                    clause_weight[c] = 1;
                else
                {
                    clause_weight[c] = 0;
                    if (clause_weight[c] > s_inc && already_in_soft_large_weight_stack[c] == 0)
                    {
                        already_in_soft_large_weight_stack[c] = 1;
                        soft_large_weight_clauses[soft_large_weight_clauses_count++] = c;
                    }
                }
            }
        }
        else // weighted not partial
        {
            for (int c = 0; c < num_clauses; c++)
            {
                already_in_soft_large_weight_stack[c] = 0;
                clause_selected_count[c] = 0;
                clause_weight[c] = tuned_org_clause_weight[c];
                if (clause_weight[c] > s_inc && already_in_soft_large_weight_stack[c] == 0)
                {
                    already_in_soft_large_weight_stack[c] = 1;
                    soft_large_weight_clauses[soft_large_weight_clauses_count++] = c;
                }
            }
        }
    }
    else // unweighted
    {
        for (int c = 0; c < num_clauses; c++)
        {
            already_in_soft_large_weight_stack[c] = 0;
            clause_selected_count[c] = 0;

            if (org_clause_weight[c] == top_clause_weight)
                clause_weight[c] = 1;
            else
            {
                if (num_hclauses > 0)
                {
                    clause_weight[c] = 0;
                }
                else
                {
                    clause_weight[c] = coe_soft_clause_weight;
                    if (clause_weight[c] > 1 && already_in_soft_large_weight_stack[c] == 0)
                    {
                        already_in_soft_large_weight_stack[c] = 1;
                        soft_large_weight_clauses[soft_large_weight_clauses_count++] = c;
                    }
                }
            }
        }
    }

    if (init_solution.size() == 0)
    {
        for (int v = 1; v <= num_vars; v++)
        {
            cur_soln[v] = rand() % 2;
            time_stamp[v] = 0;
            unsat_app_count[v] = 0;
        }
    }
    else
    {
        for (int v = 1; v <= num_vars; v++)
        {
            cur_soln[v] = init_solution[v];
            if (cur_soln[v] != 0 && cur_soln[v] != 1)
                cur_soln[v] = rand() % 2;
            time_stamp[v] = 0;
            unsat_app_count[v] = 0;
        }
    }
    local_soln_feasible = 0;
    hard_unsat_nb = 0;
    soft_unsat_weight = 0;
    hardunsat_stack_fill_pointer = 0;
    softunsat_stack_fill_pointer = 0;
    unsatvar_stack_fill_pointer = 0;
    large_weight_clauses_count = 0;

    for (int c = 0; c < num_clauses; ++c)
    {
        sat_count[c] = 0;
        for (int j = 0; j < clause_lit_count[c]; ++j)
        {
            if (cur_soln[clause_lit[c][j].var_num] == clause_lit[c][j].sense)
            {
                sat_count[c]++;
                sat_var[c] = clause_lit[c][j].var_num;
            }
        }
        if (sat_count[c] == 0)
        {
            unsat(c);
        }
    }

    for (int v = 1; v <= num_vars; v++)
    {
        score[v] = 0.0;
        for (int i = 0; i < var_lit_count[v]; ++i)
        {
            int c = var_lit[v][i].clause_num;
            if (sat_count[c] == 0)
                score[v] += clause_weight[c];
            else if (sat_count[c] == 1 && var_lit[v][i].sense == cur_soln[v])
                score[v] -= clause_weight[c];
        }
    }

    goodvar_stack_fill_pointer = 0;
    for (int v = 1; v <= num_vars; v++)
    {
        if (score[v] > 0)
        {
            already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
            mypush(v, goodvar_stack);
        }
        else
            already_in_goodvar_stack[v] = -1;
    }
}

int USW::pick_var()
{
    int i, v;
    int best_var;
    int sel_c;
    lit *p;

    if (goodvar_stack_fill_pointer > 0)
    {
        int best_array_count = 0;
        if ((rand() % MY_RAND_MAX_INT) * BASIC_SCALE < rdprob)
            return goodvar_stack[rand() % goodvar_stack_fill_pointer];

        if (goodvar_stack_fill_pointer < hd_count_threshold)
        {
            best_var = goodvar_stack[0];

            for (i = 1; i < goodvar_stack_fill_pointer; ++i)
            {
                v = goodvar_stack[i];
                if (score[v] > score[best_var])
                {
                    best_var = v;
                }
                else if (score[v] == score[best_var])
                {
                    if (time_stamp[v] < time_stamp[best_var])
                    {
                        best_var = v;
                    }
                }
            }
            return best_var;
        }
        else
        {
            best_var = goodvar_stack[rand() % goodvar_stack_fill_pointer];

            for (i = 1; i < hd_count_threshold; ++i)
            {
                v = goodvar_stack[rand() % goodvar_stack_fill_pointer];
                if (score[v] > score[best_var])
                {
                    best_var = v;
                }
                else if (score[v] == score[best_var])
                {
                    if (time_stamp[v] < time_stamp[best_var])
                    {
                        best_var = v;
                    }
                }
            }
            return best_var;
        }
    }

    update_clause_weights();

    if (hardunsat_stack_fill_pointer > 0)
    {
        sel_c = hardunsat_stack[rand() % hardunsat_stack_fill_pointer];
    }
    else
    {
        sel_c = softunsat_stack[rand() % softunsat_stack_fill_pointer];
    }
    if ((rand() % MY_RAND_MAX_INT) * BASIC_SCALE < rwprob)
        return clause_lit[sel_c][rand() % clause_lit_count[sel_c]].var_num;

    best_var = clause_lit[sel_c][0].var_num;
    p = clause_lit[sel_c];
    for (p++; (v = p->var_num) != 0; p++)
    {
        if (score[v] > score[best_var])
            best_var = v;
        else if (score[v] == score[best_var])
        {
            if (time_stamp[v] < time_stamp[best_var])
                best_var = v;
        }
    }

    return best_var;
}

int USW::nearestPowerOfTen(double num)
{
    double exponent = std::log10(num);
    int floorExponent = std::floor(exponent);
    int ceilExponent = std::ceil(exponent);
    double floorPower = std::pow(10, floorExponent);
    double ceilPower = std::pow(10, ceilExponent);
    if (num - floorPower < ceilPower - num) {
        return static_cast<int>(floorPower);
    } else {
        return static_cast<int>(ceilPower);
    }
}

long long USW::closestPowerOfTen(double num)
{
    if (num <= 5) return 1;

    int n = ceil(log10(num));
    int x = round(num / pow(10, n-1));

    if (x == 10) {
        x = 1;
        n += 1;
    }

    return pow(10, n-1) * x;
}

long long USW::floorToPowerOfTen(double x)
{
    if (x <= 0.0)
    {
        return 0;
    }
    int exponent = (int)log10(x);
    double powerOfTen = pow(10, exponent);
    long long result = (long long)powerOfTen;
    if (x < result)
    {
        result /= 10;
    }
    return result;
}

// ============================================================
// DAC Checkpoint Protocol
// ============================================================

double USW::compute_weight_mean()
{
    double sum = 0.0;
    for (int c = 0; c < num_clauses; c++)
        sum += clause_weight[c];
    return sum / (double)num_clauses;
}

double USW::compute_weight_std(double mean)
{
    double sum_sq = 0.0;
    for (int c = 0; c < num_clauses; c++)
    {
        double diff = clause_weight[c] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / (double)num_clauses);
}

void USW::dac_emit_state()
{
    double w_mean = compute_weight_mean();
    double w_std = compute_weight_std(w_mean);

    // Format: DAC_STATE <step> <hard_unsat> <soft_unsat_weight> <opt_unsat_weight>
    //         <total_step> <weight_mean> <weight_std> <feasible> <time>
    //         <num_hard_large> <num_soft_large> <goodvar_count>
    cout << "DAC_STATE"
         << " " << step
         << " " << hard_unsat_nb
         << " " << soft_unsat_weight
         << " " << opt_unsat_weight
         << " " << total_step
         << " " << w_mean
         << " " << w_std
         << " " << (hard_unsat_nb == 0 ? 1 : 0)
         << " " << get_runtime()
         << " " << large_weight_clauses_count
         << " " << soft_large_weight_clauses_count
         << " " << goodvar_stack_fill_pointer
         << endl;
    fflush(stdout);
}

void USW::dac_read_params()
{
    string cmd;
    cin >> cmd;

    if (cmd == "STOP")
    {
        if (verify_sol() == 1)
            cout << "c verified" << endl;
        print_best_solution();
        free_memory();
        exit(0);
    }
    else if (cmd == "CONTINUE")
    {
        return;
    }
    else
    {
        // Parse 6 parameters: h_inc s_inc smooth_prob soft_smooth_prob rwprob rdprob
        double new_h_inc = atof(cmd.c_str());
        double new_s_inc;
        float new_smooth_prob, new_soft_smooth_prob, new_rwprob, new_rdprob;

        cin >> new_s_inc >> new_smooth_prob >> new_soft_smooth_prob >> new_rwprob >> new_rdprob;

        h_inc = new_h_inc;
        s_inc = new_s_inc;
        smooth_probability = new_smooth_prob;
        soft_smooth_probability = new_soft_smooth_prob;
        rwprob = new_rwprob;
        rdprob = new_rdprob;
    }
}

// ============================================================
// Main search loop (with DAC checkpoint support)
// ============================================================

void USW::local_search_with_decimation(char *inputfile)
{
    if (1 == problem_weighted)
    {
        if (0 != num_hclauses) // weighted partial
        {
            coe_tuned_weight = 1.0/(double)floorToPowerOfTen(double(top_clause_weight - 1) / (double)(num_sclauses));

            for (int c = 0; c < num_clauses; c++)
            {
                if (org_clause_weight[c] != top_clause_weight)
                {
                    tuned_org_clause_weight[c] = (double)org_clause_weight[c] * coe_tuned_weight;
                }
            }
        }
        else // weighted not partial
        {
            softclause_weight_threshold = 0;
            soft_smooth_probability = 1E-3;
            hd_count_threshold = 22;
            rdprob = 0.036;
            rwprob = 0.48;
            s_inc = 1.0;

            coe_tuned_weight = ((double)coe_soft_clause_weight)/floorToPowerOfTen((double(top_clause_weight - 1) / (double)(num_sclauses)));

            cout << "c coe_tuned_weight: " << coe_tuned_weight << endl;
            for (int c = 0; c < num_clauses; c++)
            {
                tuned_org_clause_weight[c] = (double)org_clause_weight[c] * coe_tuned_weight;
            }
        }
    }
    else
    {
        if (0 == num_hclauses)  // unweighted not partial
        {
            hd_count_threshold = 94;
            coe_soft_clause_weight = 397;
            rdprob = 0.007;
            rwprob = 0.047;
            soft_smooth_probability = 0.002;
            softclause_weight_threshold = 550;
        }
    }
    Decimation deci(var_lit, var_lit_count, clause_lit, org_clause_weight, top_clause_weight);
    deci.make_space(num_clauses, num_vars);

    // In DAC mode, emit initial state before search begins
    if (dac_mode)
    {
        cout << "DAC_READY " << num_vars << " " << num_clauses << " "
             << num_hclauses << " " << num_sclauses << " " << total_soft_weight << endl;
        fflush(stdout);
    }

#ifdef _WIN32
    opt_unsat_weight = _I64_MAX;
#else
    opt_unsat_weight = __LONG_LONG_MAX__;
#endif

    for (tries = 1; tries < max_tries; ++tries)
    {
        deci.init(local_opt_soln, best_soln, unit_clause, unit_clause_count, clause_lit_count);
        deci.unit_prosess();
        init(deci.fix);

        long long local_opt;
#ifdef _WIN32
        local_opt = _I64_MAX;
#else
        local_opt = __LONG_LONG_MAX__;
#endif
        max_flips = max_non_improve_flip;

        // DAC: emit state at start of each try
        if (dac_mode)
        {
            dac_emit_state();
            dac_read_params();
        }

        unsigned int dac_flip_counter = 0;

        for (step = 1; step < max_flips; ++step)
        {
            if (hard_unsat_nb == 0)
            {
                local_soln_feasible = 1;
                if (local_opt > soft_unsat_weight)
                {
                    local_opt = soft_unsat_weight;
                    max_flips = step + max_non_improve_flip;
                }
                if (soft_unsat_weight < opt_unsat_weight)
                {
                    opt_time = get_runtime();
                    cout << "o " << soft_unsat_weight << " " << opt_time << endl;
                    opt_unsat_weight = soft_unsat_weight;

                    for (int v = 1; v <= num_vars; ++v)
                        best_soln[v] = cur_soln[v];
                    if (opt_unsat_weight <= best_known || 0 == opt_unsat_weight)
                    {
                        cout << "c best solution found." << endl;
                        if (opt_unsat_weight < best_known)
                            cout << "c a better solution " << opt_unsat_weight << endl;

                        if (dac_mode)
                        {
                            dac_emit_state();
                            dac_read_params();
                        }
                        return;
                    }
                }
                if (best_soln_feasible == 0)
                {
                    best_soln_feasible = 1;
                }
            }

            int flipvar = pick_var();
            flip(flipvar);
            time_stamp[flipvar] = step;
            total_step++;

            // DAC checkpoint: emit state and read params every N flips
            if (dac_mode)
            {
                dac_flip_counter++;
                if (dac_flip_counter >= (unsigned int)dac_checkpoint_interval)
                {
                    dac_flip_counter = 0;
                    dac_emit_state();
                    dac_read_params();
                }
            }
        }
    }
}

void USW::hard_increase_weights()
{
    int i, c, v;
    for (i = 0; i < hardunsat_stack_fill_pointer; ++i)
    {
        c = hardunsat_stack[i];
        clause_weight[c] += h_inc;

        for (lit *p = clause_lit[c]; (v = p->var_num) != 0; p++)
        {
            score[v] += h_inc;
            if (score[v] > 0 && already_in_goodvar_stack[v] == -1)
            {
                already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                mypush(v, goodvar_stack);
            }
        }
    }
    return;
}

void USW::soft_increase_weights_partial()
{
    int i, c, v;

    if (1 == problem_weighted)
    {
        for (i = 0; i < num_sclauses; ++i)
        {
            c = soft_clause_num_index[i];
            clause_weight[c] += tuned_org_clause_weight[c];
            if (sat_count[c] <= 0) // unsat
            {
                for (lit *p = clause_lit[c]; (v = p->var_num) != 0; p++)
                {
                    score[v] += tuned_org_clause_weight[c];
                    if (score[v] > 0 && already_in_goodvar_stack[v] == -1)
                    {
                        already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                        mypush(v, goodvar_stack);
                    }
                }
            }
            else if (sat_count[c] < 2) // sat
            {
                for (lit *p = clause_lit[c]; (v = p->var_num) != 0; p++)
                {
                    if (p->sense == cur_soln[v])
                    {
                        score[v] -= tuned_org_clause_weight[c];
                        if (score[v] <= 0 && -1 != already_in_goodvar_stack[v])
                        {
                            int index = already_in_goodvar_stack[v];
                            int last_v = mypop(goodvar_stack);
                            goodvar_stack[index] = last_v;
                            already_in_goodvar_stack[last_v] = index;
                            already_in_goodvar_stack[v] = -1;
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (i = 0; i < num_sclauses; ++i)
        {
            c = soft_clause_num_index[i];
            clause_weight[c] += s_inc;

            if (sat_count[c] <= 0) // unsat
            {
                for (lit *p = clause_lit[c]; (v = p->var_num) != 0; p++)
                {
                    score[v] += s_inc;
                    if (score[v] > 0 && already_in_goodvar_stack[v] == -1)
                    {
                        already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                        mypush(v, goodvar_stack);
                    }
                }
            }
            else if (sat_count[c] < 2) // sat
            {
                for (lit *p = clause_lit[c]; (v = p->var_num) != 0; p++)
                {
                    if (p->sense == cur_soln[v])
                    {
                        score[v] -= s_inc;
                        if (score[v] <= 0 && -1 != already_in_goodvar_stack[v])
                        {
                            int index = already_in_goodvar_stack[v];
                            int last_v = mypop(goodvar_stack);
                            goodvar_stack[index] = last_v;
                            already_in_goodvar_stack[last_v] = index;
                            already_in_goodvar_stack[v] = -1;
                        }
                    }
                }
            }
        }
    }
    return;
}

void USW::soft_increase_weights_not_partial()
{
    int i, c, v;

    if (1 == problem_weighted)
    {
        for (i = 0; i < softunsat_stack_fill_pointer; ++i)
        {
            c = softunsat_stack[i];
            if (clause_weight[c] >= tuned_org_clause_weight[c] + softclause_weight_threshold)
                continue;
            else
                clause_weight[c] += s_inc;

            if (clause_weight[c] > s_inc && already_in_soft_large_weight_stack[c] == 0)
            {
                already_in_soft_large_weight_stack[c] = 1;
                soft_large_weight_clauses[soft_large_weight_clauses_count++] = c;
            }
            for (lit *p = clause_lit[c]; (v = p->var_num) != 0; p++)
            {
                score[v] += s_inc;
                if (score[v] > 0 && already_in_goodvar_stack[v] == -1)
                {
                    already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                    mypush(v, goodvar_stack);
                }
            }
        }
    }
    else
    {
        for (i = 0; i < softunsat_stack_fill_pointer; ++i)
        {
            c = softunsat_stack[i];
            if (clause_weight[c] >= coe_soft_clause_weight + softclause_weight_threshold)
                continue;
            else
                clause_weight[c] += s_inc;

            if (clause_weight[c] > s_inc && already_in_soft_large_weight_stack[c] == 0)
            {
                already_in_soft_large_weight_stack[c] = 1;
                soft_large_weight_clauses[soft_large_weight_clauses_count++] = c;
            }
            for (lit *p = clause_lit[c]; (v = p->var_num) != 0; p++)
            {
                score[v] += s_inc;
                if (score[v] > 0 && already_in_goodvar_stack[v] == -1)
                {
                    already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                    mypush(v, goodvar_stack);
                }
            }
        }
    }
    return;
}

void USW::hard_smooth_weights()
{
    int i, clause, v;
    for (i = 0; i < large_weight_clauses_count; i++)
    {
        clause = large_weight_clauses[i];
        if (sat_count[clause] > 0)
        {
            clause_weight[clause] -= h_inc;

            if (clause_weight[clause] == 1)
            {
                large_weight_clauses[i] = large_weight_clauses[--large_weight_clauses_count];
                i--;
            }
            if (sat_count[clause] == 1)
            {
                v = sat_var[clause];
                score[v] += h_inc;
                if (score[v] > 0 && already_in_goodvar_stack[v] == -1)
                {
                    already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                    mypush(v, goodvar_stack);
                }
            }
        }
    }
    return;
}

void USW::soft_smooth_weights()
{
    int i, clause, v;

    for (i = 0; i < soft_large_weight_clauses_count; i++)
    {
        clause = soft_large_weight_clauses[i];
        if (sat_count[clause] > 0)
        {
            clause_weight[clause] -= s_inc;
            if (clause_weight[clause] <= s_inc && already_in_soft_large_weight_stack[clause] == 1)
            {
                already_in_soft_large_weight_stack[clause] = 0;
                soft_large_weight_clauses[i] = soft_large_weight_clauses[--soft_large_weight_clauses_count];
                i--;
            }
            if (sat_count[clause] == 1)
            {
                v = sat_var[clause];
                score[v] += s_inc;
                if (score[v] > 0 && already_in_goodvar_stack[v] == -1)
                {
                    already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                    mypush(v, goodvar_stack);
                }
            }
        }
    }
    return;
}

void USW::update_clause_weights()
{
    if (num_hclauses > 0) // partial
    {
        hard_increase_weights();
        if (0 == hard_unsat_nb)
        {
            soft_increase_weights_partial();
        }
    }
    else  // not partial
    {
        if (((rand() % MY_RAND_MAX_INT) * BASIC_SCALE) < soft_smooth_probability && soft_large_weight_clauses_count > soft_large_clause_count_threshold)
        {
            soft_smooth_weights();
        }
        else
        {
            soft_increase_weights_not_partial();
        }
    }
}

void USW::update_goodvarstack1(int flipvar)
{
    int v;
    for (int index = goodvar_stack_fill_pointer - 1; index >= 0; index--)
    {
        v = goodvar_stack[index];
        if (score[v] <= 0)
        {
            goodvar_stack[index] = mypop(goodvar_stack);
            already_in_goodvar_stack[v] = -1;
        }
    }

    for (int i = 0; i < var_neighbor_count[flipvar]; ++i)
    {
        v = var_neighbor[flipvar][i];
        if (score[v] > 0)
        {
            if (already_in_goodvar_stack[v] == -1)
            {
                already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                mypush(v, goodvar_stack);
            }
        }
    }
}

void USW::update_goodvarstack2(int flipvar)
{
    if (score[flipvar] > 0 && already_in_goodvar_stack[flipvar] == -1)
    {
        already_in_goodvar_stack[flipvar] = goodvar_stack_fill_pointer;
        mypush(flipvar, goodvar_stack);
    }
    else if (score[flipvar] <= 0 && already_in_goodvar_stack[flipvar] != -1)
    {
        int index = already_in_goodvar_stack[flipvar];
        int last_v = mypop(goodvar_stack);
        goodvar_stack[index] = last_v;
        already_in_goodvar_stack[last_v] = index;
        already_in_goodvar_stack[flipvar] = -1;
    }
    int i, v;
    for (i = 0; i < var_neighbor_count[flipvar]; ++i)
    {
        v = var_neighbor[flipvar][i];
        if (score[v] > 0)
        {
            if (already_in_goodvar_stack[v] == -1)
            {
                already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                mypush(v, goodvar_stack);
            }
        }
        else if (already_in_goodvar_stack[v] != -1)
        {
            int index = already_in_goodvar_stack[v];
            int last_v = mypop(goodvar_stack);
            goodvar_stack[index] = last_v;
            already_in_goodvar_stack[last_v] = index;
            already_in_goodvar_stack[v] = -1;
        }
    }
}

void USW::flip(int flipvar)
{
    int i, v, c;
    int index;
    lit *clause_c;

    double org_flipvar_score = score[flipvar];
    cur_soln[flipvar] = 1 - cur_soln[flipvar];

    for (i = 0; i < var_lit_count[flipvar]; ++i)
    {
        c = var_lit[flipvar][i].clause_num;
        clause_c = clause_lit[c];

        if (cur_soln[flipvar] == var_lit[flipvar][i].sense)
        {
            ++sat_count[c];
            if (sat_count[c] == 2)
            {
                score[sat_var[c]] += clause_weight[c];
                if (score[sat_var[c]] > 0 && -1 == already_in_goodvar_stack[sat_var[c]])
                {
                    already_in_goodvar_stack[sat_var[c]] = goodvar_stack_fill_pointer;
                    mypush(sat_var[c], goodvar_stack);
                }
            }
            else if (sat_count[c] == 1)
            {
                sat_var[c] = flipvar;
                for (lit *p = clause_c; (v = p->var_num) != 0; p++)
                {
                    score[v] -= clause_weight[c];
                    if (score[v] <= 0 && -1 != already_in_goodvar_stack[v])
                    {
                        int index = already_in_goodvar_stack[v];
                        int last_v = mypop(goodvar_stack);
                        goodvar_stack[index] = last_v;
                        already_in_goodvar_stack[last_v] = index;
                        already_in_goodvar_stack[v] = -1;
                    }
                }
                sat(c);
            }
        }
        else
        {
            --sat_count[c];
            if (sat_count[c] == 1)
            {
                for (lit *p = clause_c; (v = p->var_num) != 0; p++)
                {
                    if (p->sense == cur_soln[v])
                    {
                        score[v] -= clause_weight[c];
                        if (score[v] <= 0 && -1 != already_in_goodvar_stack[v])
                        {
                            int index = already_in_goodvar_stack[v];
                            int last_v = mypop(goodvar_stack);
                            goodvar_stack[index] = last_v;
                            already_in_goodvar_stack[last_v] = index;
                            already_in_goodvar_stack[v] = -1;
                        }
                        sat_var[c] = v;
                        break;
                    }
                }
            }
            else if (sat_count[c] == 0)
            {
                for (lit *p = clause_c; (v = p->var_num) != 0; p++)
                {
                    score[v] += clause_weight[c];
                    if (score[v] > 0 && -1 == already_in_goodvar_stack[v])
                    {
                        already_in_goodvar_stack[v] = goodvar_stack_fill_pointer;
                        mypush(v, goodvar_stack);
                    }
                }
                unsat(c);
            }
        }
    }

    score[flipvar] = -org_flipvar_score;
    if (score[flipvar] > 0 && already_in_goodvar_stack[flipvar] == -1)
    {
        already_in_goodvar_stack[flipvar] = goodvar_stack_fill_pointer;
        mypush(flipvar, goodvar_stack);
    }
    else if (score[flipvar] <= 0 && already_in_goodvar_stack[flipvar] != -1)
    {
        int index = already_in_goodvar_stack[flipvar];
        int last_v = mypop(goodvar_stack);
        goodvar_stack[index] = last_v;
        already_in_goodvar_stack[last_v] = index;
        already_in_goodvar_stack[flipvar] = -1;
    }
}

void USW::print_best_solution()
{
    if (best_soln_feasible == 0)
        return;

    printf("v ");
    for (int i = 1; i <= num_vars; i++)
    {
        if (best_soln[i] == 0)
            printf("0");
        else
            printf("1");
    }
    printf("\n");
}

bool USW::verify_sol()
{
    int c, j, flag;
    long long verify_unsat_weight = 0;

    for (c = 0; c < num_clauses; ++c)
    {
        flag = 0;
        for (j = 0; j < clause_lit_count[c]; ++j)
            if (best_soln[clause_lit[c][j].var_num] == clause_lit[c][j].sense)
            {
                flag = 1;
                break;
            }

        if (flag == 0)
        {
            if (org_clause_weight[c] == top_clause_weight)
            {
                cout << "c Error: hard clause " << c << " is not satisfied" << endl;
                return 0;
            }
            else
            {
                verify_unsat_weight += org_clause_weight[c];
            }
        }
    }

    if (verify_unsat_weight == opt_unsat_weight)
        return 1;
    else
    {
        cout << "c Error: find opt=" << opt_unsat_weight << ", but verified opt=" << verify_unsat_weight << endl;
    }
    return 0;
}

void USW::simple_print()
{
    if (best_soln_feasible != 0)
    {
        if (verify_sol() == 1)
            cout << opt_unsat_weight << '\t' << opt_time << endl;
        else
            cout << "solution is wrong " << endl;
    }
    else
        cout << -1 << '\t' << -1 << endl;
}

inline void USW::unsat(int clause)
{
    if (org_clause_weight[clause] == top_clause_weight)
    {
        index_in_hardunsat_stack[clause] = hardunsat_stack_fill_pointer;
        mypush(clause, hardunsat_stack);
        hard_unsat_nb++;
    }
    else
    {
        index_in_softunsat_stack[clause] = softunsat_stack_fill_pointer;
        mypush(clause, softunsat_stack);
        soft_unsat_weight += org_clause_weight[clause];
    }
}

inline void USW::sat(int clause)
{
    int index, last_unsat_clause;

    if (org_clause_weight[clause] == top_clause_weight)
    {
        last_unsat_clause = mypop(hardunsat_stack);
        index = index_in_hardunsat_stack[clause];
        hardunsat_stack[index] = last_unsat_clause;
        index_in_hardunsat_stack[last_unsat_clause] = index;
        hard_unsat_nb--;
    }
    else
    {
        last_unsat_clause = mypop(softunsat_stack);
        index = index_in_softunsat_stack[clause];
        softunsat_stack[index] = last_unsat_clause;
        index_in_softunsat_stack[last_unsat_clause] = index;
        soft_unsat_weight -= org_clause_weight[clause];
    }
}

#endif
