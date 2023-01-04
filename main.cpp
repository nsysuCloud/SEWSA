#include <iostream>
#include <libgen.h>
#include <cstring>
#include <iomanip>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <fstream>
#include <bitset>
#include <sstream>
#include <algorithm>

using namespace std;

typedef vector<int> i1d;
typedef vector<i1d> i2d;
typedef vector<i2d> i3d;
typedef vector<i3d> i4d;
typedef vector<double> d1d;
typedef vector<d1d> d2d;
typedef vector<d2d> d3d;
typedef struct Task
{
	int i, j;
	int num_resource; //被哪個資源執行
	double start;	  //開始執行時間
	double end;		  //結束執行時間
} Task;

typedef struct Resource
{
	vector<Task> WorkSet; //工作集
	double execute_time;  //執行總時間
} Resource;

// 0.0 testing parameters
int two_opt_eval = 0;
vector<clock_t> time_start;
vector<clock_t> time_end;
d1d part_time;

// 0.1 environment settings
int num_run;
int num_iter;
int num_bit_sol;
int num_searcher;
int num_region;
int num_sample;
int num_player;
char cost_time;
int num_evaluation;

// 0.2 search results
d1d avg_obj_value_iter;
double best_obj_value;
i1d best_sol;
double overall_best_obj_value;
i1d overall_best_sol;

// 0.3 search algorithm
i2d searcher_sol;	 // [searcher, num_bit_sol]
i3d sample_sol;		 // [region, sample, num_bit_sol]
i2d sample_sol_best; // [region, num_bit_sol]
i4d sampleV_sol;	 // [searcher, region, sample, num_bit_sol]
i4d sampleV_sol_id_bit;

d1d searcher_sol_fitness;
d2d sample_sol_fitness;
d1d sample_sol_best_fitness;
d3d sampleV_sol_fitness;

i1d searcher_region_id; // [searcher], region to which a searcher is assigned

int num_identity_bit; // number of tabu bits
i2d identity_bit;	  // region id: tabu bits

d1d region_it;
d1d region_hl;

d2d expected_value;
d1d T_j;
d2d V_ij;
d1d M_j;

void run();

// 0.4 read dataset
int num_machine; // number of machine
i1d num_task;	 // number of task
d1d Crent;		 // the rent of machine
d2d Ctrans;		 // traslational cost of each machine
d2d Ttrans;		 // traslational cost of each task
d2d TDataSize;	 // the data size of each sub-task
d3d Texe;		 // excuted time of each sub task on each machine
ifstream fin;
ofstream fout1;
ofstream fout2;

void read_dataset();
void reset_data();
vector<Resource> resource;
vector<vector<Task>> all_task;
i1d searcher_repeat;
int count_evaluation;
int inc;
d1d eval_best;

// 1. initialization
void init();

// 2. resource arrangement
void resource_arrangement();
void region_identity_bit();
void assign_search_region(int search_idx, int region_idx);

i1d rand_sol();

// 3. vision search
void vision_search();
void transit();
void compute_expected_value();
double evaluate_fitness(const i1d &sol);
void vision_selection(int player);
double two_opt(i1d &sol, double best_result);

// 4. marketing survey
void marketing_survey();

void check_sol(const i1d &sol);

int main(int argc, char **argv)
{

	srand((unsigned)time(NULL));

	num_run = atoi(argv[1]);
	num_evaluation = atoi(argv[2]);
	num_searcher = atoi(argv[3]);
	num_region = atoi(argv[4]);
	num_sample = atoi(argv[5]);
	num_player = atoi(argv[6]);
	cost_time = *argv[7];
	string dataname = argv[8];
	string outputname = argv[9];
	fin.open(dataname);

	if (!fin)
	{
		cout << "Reading file is fail!" << endl;
	}
	// fout1.open("result_iter.txt");
	// if(!fout1){cout<<"Writing file is fail!"<<endl;}
	fout2.open(outputname);
	if (!fout2)
	{
		cout << "Writing file is fail!" << endl;
	}

	cout << "dataname = " << dataname << endl;
	cout << "searcher = " << num_searcher << endl;
	cout << "region = " << num_region << endl;
	cout << "sample = " << num_sample << endl;
	cout << "player = " << num_player << endl;

	read_dataset();
	cout << "start..." << endl;
	clock_t begin = clock();
	run();
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "time: " << elapsed_secs << " sec." << endl;
	cout << "average_time: " << elapsed_secs / num_run << " sec." << endl;
	cout << "2opt_evaluation: " << two_opt_eval << endl;
	cout << "count_evaluation: " << count_evaluation << endl;
	cout << "part_init: " << part_time[0] << endl;
	cout << "part_SEWSA: " << part_time[1] << endl;
	cout << "part_2opt: " << part_time[2] << endl;
	cout << "done." << endl;

	// fout1.close();
	fout2.close();

	return 0;
}

void run()
{
	avg_obj_value_iter.assign(num_iter, 0.0);
	overall_best_obj_value = 100000.0;

	d1d last_eval_best;
	last_eval_best.assign(num_run, 100000.0);
	double std = 0.0;

	for (int i = 0; i < num_evaluation; i++)
		eval_best.push_back(0);

	time_start.assign(3, 0);
	time_end.assign(3, 0);
	part_time.assign(3, 0);

	for (int r = 0; r < num_run; r++)
	{
		// cout<<"run = "<<r<<endl;
		time_start[0] = clock();
		init(); // 1. initialization
		time_end[0] = clock();
		part_time[0] = double(time_end[0] - time_start[0]) / CLOCKS_PER_SEC;

		time_start[1] = clock();
		resource_arrangement(); // 2. resource arrangement
		while (count_evaluation <= num_evaluation)
		{
			inc = count_evaluation;
			// cout<<"\niter : "<< i <<endl;
			vision_search();	// 3. vision search
			marketing_survey(); // 4. marketing survey

			if (count_evaluation < num_evaluation)
				for (int i = inc; i < count_evaluation; i++)
					eval_best[i] += best_obj_value;
			else
				for (int i = inc; i < num_evaluation; i++)
					eval_best[i] += best_obj_value;
		}
		last_eval_best[r] = best_obj_value;
		time_end[1] = clock();
		part_time[1] = double(time_end[1] - time_start[1]) / CLOCKS_PER_SEC;
	}

	// for (int i = 0; i < num_iter; i++){
	// cout << i <<": "<< avg_obj_value_iter[i]/num_run << endl;
	// fout1 << avg_obj_value_iter[i]/num_run <<endl;
	//}

	double mean;
	for (int i = 1; i < eval_best.size(); i++)
	{
		fout2 << eval_best[i] / num_run << endl;
		if (i == eval_best.size() - 1)
		{
			mean = eval_best[i] / num_run;
		}
	}

	// calculate standard deviation
	for (int i = 0; i < num_run; i++)
	{
		cout << "run: " << i << " answer: " << last_eval_best[i] << endl;
		std += pow(last_eval_best[i] - mean, 2);
	}
	std /= (num_run - 1);
	std = sqrt(std);

	fout2 << "best makespan: " << overall_best_obj_value << endl;
	fout2 << "\nstandard deviation: " << std << endl;
	cout << "\nstandard deviation: " << std << endl;
	// for(int i=0;i<num_bit_sol;i++)
	//	cout<<overall_best_sol[i]<<" ";

	cout << "\nbest_obj_value: " << overall_best_obj_value << endl;
	cout << "\navg_obj_value: " << eval_best[eval_best.size() - 1] / num_run << endl;
}

// 1. initialization
void init() /*@\label{se:init:begin}@*/
{
	// set aside arrays for searchers, samples, and sampleV (the
	// crossover results of searchers and samples)
	searcher_sol.assign(num_searcher, i1d(num_bit_sol, 0));
	sample_sol.assign(num_region, i2d(num_sample, i1d(num_bit_sol, 0)));
	sample_sol_best.assign(num_region, i1d(num_bit_sol, 0));
	sampleV_sol.assign(num_searcher, i3d(num_region, i2d(num_sample * 2, i1d(num_bit_sol, 0))));

	searcher_sol_fitness.assign(num_searcher, 0.0);
	sample_sol_fitness.assign(num_region, d1d(num_sample, 0.0));
	sample_sol_best_fitness.assign(num_region, 100000.0);
	sampleV_sol_fitness.assign(num_searcher, d2d(num_region, d1d(num_sample * 2, 0.0)));

	best_sol.assign(num_bit_sol, 0);

	searcher_repeat.assign(num_searcher, -1);
	count_evaluation = 0;

	all_task.resize(TDataSize.size());
	resource.resize(num_machine);
	for (int i = 0; i < resource.size(); i++)
	{
		int remainder;
		if (i < num_bit_sol % num_machine)
			remainder = 1;
		else
			remainder = 0;

		resource[i].WorkSet.resize(num_bit_sol / num_machine + remainder);
	}
	reset_data();
	for (int i = 0; i < num_searcher; i++)
		searcher_sol[i] = rand_sol();

	best_obj_value = 100000.0;

} /*@\label{se:init:end}@*/

// 2. resource arrangement
void resource_arrangement() /*@\label{se:ra:begin}@*/
{
	// 2.1. initialize searchers and regions
	num_identity_bit = 1; /*@\label{se:2.1b}@*/
	searcher_region_id.assign(num_searcher, 0);
	identity_bit.assign(num_region, i1d(num_identity_bit, 0));
	region_identity_bit(); /*@\label{se:2.1e}@*/
	sampleV_sol_id_bit.assign(num_searcher, i3d(num_region, i2d(num_sample * 2, i1d(num_identity_bit, 0))));

	// 2.1.1 assign searcher to its region and their investment
	for (int i = 0; i < num_searcher; i++)		 /*@\label{se:2.1.1b}@*/
		assign_search_region(i, i % num_region); /*@\label{se:2.1.1e}@*/

	// 2.1.2 initialize the sample solutions
	for (int i = 0; i < num_region; i++)
	{ /*@\label{se:2.1.2b}@*/
		for (int j = 0; j < num_sample; j++)
		{
			sample_sol[i][j] = rand_sol();

			int factor = num_task.size() / num_region;
			int R = (rand() % factor) * num_region + identity_bit[i][0];
			while (R >= num_task.size())
				R = (rand() % factor) * num_region + identity_bit[i][0];

			if (sample_sol[i][j][0] != R)
			{
				for (int k = 1; k < num_bit_sol; k++)
				{
					if (sample_sol[i][j][k] == R)
					{
						sample_sol[i][j][k] = sample_sol[i][j][0];
						sample_sol[i][j][0] = R;
						break;
					}
				}
			}

		} /*@\label{se:2.1.2e}@*/
	}

	// 2.2. initialize the investment and set how long regions have not been searched
	region_it.assign(num_region, 0.0); /*@\label{se:2.2b}@*/
	region_hl.assign(num_region, 1.0);
	for (int i = 0; i < num_searcher; i++)
	{
		int idx = searcher_region_id[i];
		region_it[idx]++;
		region_hl[idx] = 1.0;
	} /*@\label{se:2.2e}@*/

	// 2.3. initialize the expected values (ev)
	expected_value.assign(num_searcher, d1d(num_region, 0.0)); /*@\label{se:2.3b}@*/
	T_j.assign(num_region, 0.0);
	V_ij.assign(num_searcher, d1d(num_region, 0.0));
	M_j.assign(num_region, 0.0); /*@\label{se:2.3e}@*/
}

// subfunction: 2.1. initialize searchers and regions
void region_identity_bit()
{
	for (int i = 0; i < num_region; i++)
	{
		int j = num_identity_bit;
		identity_bit[i][--j] = i;
		/*
	  for (int k = 0; k < num_identity_bit; k++)
	  cout << identity_bit[i][k];
	  cout << endl;
	*/
	}
}

// subfunction: 2.1.1 assign searchers to regions
void assign_search_region(int search_idx, int region_idx)
{
	searcher_region_id[search_idx] = region_idx;

	int factor = num_task.size() / num_region;
	int R = (rand() % factor) * num_region + region_idx;
	while (R >= num_task.size())
		R = (rand() % factor) * num_region + region_idx;

	if (searcher_sol[search_idx][0] != R)
	{
		for (int i = 1; i < num_bit_sol; i++)
		{
			if (searcher_sol[search_idx][i] == R)
			{
				searcher_sol[search_idx][i] = searcher_sol[search_idx][0];
				searcher_sol[search_idx][0] = R;
				break;
			}
		}
	}
} /*@\label{se:ra:end}@*/

// 3. vision search
void vision_search() /*@\label{se:vs:begin}@*/
{
	// 3.1 construct V (searcher X sample)
	transit();

	// 3.2 compute the expected value of all regions of searchers
	compute_expected_value();

	// 3.3 select region to which a searcher will be assigned
	vision_selection(num_player);
}

// 3.1 construct V (searcher X sample)
void transit()
{
	// 3.1.1 exchange information between searchers and samples; save
	// the results in sampleV_sol, by using the concept of LOX

	for (int i = 0; i < num_searcher; i++)
	{
		for (int j = 0; j < num_region; j++)
		{
			for (int k = 0; k < num_sample; k++)
			{
				const int m = k << 1;
				// cout << j << ", ";

				int factor = num_task.size() / num_region;
				int R1 = (rand() % factor) * num_region + identity_bit[j][0];
				int R2 = (rand() % factor) * num_region + identity_bit[j][0];
				while (R1 >= num_task.size())
					R1 = (rand() % factor) * num_region + identity_bit[j][0];
				while (R2 >= num_task.size() || R2 == R1)
					R2 = (rand() % factor) * num_region + identity_bit[j][0];
				sampleV_sol_id_bit[i][j][m][0] = R1;
				sampleV_sol_id_bit[i][j][m + 1][0] = R2;
				// cout<<"region "<<identity_bit[j][0]<<" &sampleV_sol_id_bit:"<<sampleV_sol_id_bit[i][j][m][l]<<" "<<sampleV_sol_id_bit[i][j][m+1][l]<<endl;

				// CX
				i1d tmp_sol;
				tmp_sol.assign(num_bit_sol, -1);
				int cx_point = rand() % num_bit_sol;		 // start point
				int start_value = searcher_sol[i][cx_point]; // start value
				int cx_num = sample_sol[j][k][cx_point];	 // record cycle
				// tmp_sol[cx_point] = searcher_sol[i][cx_point];

				i1d count_sol1;
				i1d count_sol2;
				count_sol1.assign(num_task.size(), 0);
				count_sol2.assign(num_task.size(), 0);
				// count_sol1[ searcher_sol[i][cx_point] ]++;
				for (int l = 0; l < num_bit_sol; l++)
				{
					count_sol1[searcher_sol[i][cx_point]]++;
					count_sol2[sample_sol[j][k][cx_point]]++;
					cx_num = sample_sol[j][k][cx_point];
					tmp_sol[cx_point] = searcher_sol[i][cx_point];
					for (int x = 0; x < num_bit_sol; x++)
					{
						int y = num_bit_sol - 1 - x;
						if (searcher_sol[i][y] == cx_num && tmp_sol[y] == -1)
						{
							cx_point = y;

							// cout<<"x="<<x;
							break;
						}
					}

					if (l >= num_bit_sol / 3 && count_sol1 == count_sol2)
						break;
				}

				// if(count_sol1 == count_sol2) cout<<"yes,";
				// else cout<<"Nooo~...";
				// cout<<endl;
				for (int l = 0; l < num_bit_sol; l++)
				{
					if (tmp_sol[l] == -1)
					{
						sampleV_sol[i][j][m][l] = sample_sol[j][k][l];
						sampleV_sol[i][j][m + 1][l] = searcher_sol[i][l];
					}
					else
					{
						sampleV_sol[i][j][m][l] = searcher_sol[i][l];
						sampleV_sol[i][j][m + 1][l] = sample_sol[j][k][l];
					}
				}
				/*
				//
				int coun1=0;
				int coun2=0;
				for(int l=0; l<num_bit_sol;l++){
					if(sampleV_sol[i][j][m][l]!=searcher_sol[i][l])	coun1++;
					if(sampleV_sol[i][j][m+1][l]!=sample_sol[j][k][l]) coun2++;
				}
				cout<<"c1 different element:"<<coun1<<"/"<<num_bit_sol<<endl;
				cout<<"c2 different element:"<<coun2<<"/"<<num_bit_sol<<endl;*/
			}
		}
	}

	// 3.1.2 mutate one bit of each solution in sampleV_sol,
	// repair the identiy bit of sampleV to be the mutaiton
	for (int i = 0; i < num_searcher; i++)
	{
		for (int j = 0; j < num_region; j++)
		{
			for (int k = 0; k < 2 * num_sample; k++)
			{
				if (sampleV_sol[i][j][k][0] != sampleV_sol_id_bit[i][j][k][0])
				{
					for (int x = 1; x < num_bit_sol; x++)
					{
						if (sampleV_sol[i][j][k][x] == sampleV_sol_id_bit[i][j][k][0])
						{
							sampleV_sol[i][j][k][x] = sampleV_sol[i][j][k][0];
							sampleV_sol[i][j][k][0] = sampleV_sol_id_bit[i][j][k][0];
						}
					}
				}
				else
				{
					int rand_num1 = rand() % (num_bit_sol - 1) + 1;
					int rand_num2 = rand() % (num_bit_sol - 1) + 1;
					while (rand_num1 == rand_num2)
						rand_num2 = rand() % (num_bit_sol - 1) + 1;

					int tmp = sampleV_sol[i][j][k][rand_num1];
					sampleV_sol[i][j][k][rand_num1] = sampleV_sol[i][j][k][rand_num2];
					sampleV_sol[i][j][k][rand_num2] = tmp;
				}
			}
		}
	}
}

// 3.2 compute the expected value for all regions of searchers
void compute_expected_value()
{
	// 3.2.1 fitness value of searchers
	for (int i = 0; i < num_searcher; i++)
		searcher_sol_fitness[i] = evaluate_fitness(searcher_sol[i]);

	// 3.2.2 fitness value of samples, M_j
	double all_sample_fitness = 0.0; // f(m_j)
	for (int i = 0; i < num_region; i++)
	{
		double rbj = sample_sol_best_fitness[i];
		int idx = -1;

		for (int j = 0; j < num_sample; j++)
		{
			sample_sol_fitness[i][j] = evaluate_fitness(sample_sol[i][j]);
			all_sample_fitness += sample_sol_fitness[i][j];
			// update fbj
			if (sample_sol_fitness[i][j] < rbj)
			{
				idx = j;
				rbj = sample_sol_fitness[i][j];
			}
		}

		if (idx >= 0)
		{
			sample_sol_best_fitness[i] = rbj;
			sample_sol_best[i] = sample_sol[i][idx];
		}
	}

	// M_j
	for (int i = 0; i < num_region; i++)
		M_j[i] = sample_sol_best_fitness[i] / all_sample_fitness;

	// 3.2.3 fitness value of sampleV, V_ij
	for (int i = 0; i < num_searcher; i++)
	{
		for (int j = 0; j < num_region; j++)
		{
			V_ij[i][j] = 0.0;
			for (int k = 0; k < num_sample; k++)
			{
				const int m = k << 1;
				sampleV_sol_fitness[i][j][m] = evaluate_fitness(sampleV_sol[i][j][m]);
				sampleV_sol_fitness[i][j][m + 1] = evaluate_fitness(sampleV_sol[i][j][m + 1]);
				V_ij[i][j] += sampleV_sol_fitness[i][j][m] + sampleV_sol_fitness[i][j][m + 1];
			}
			V_ij[i][j] /= 2 * num_sample; // *** Bugfix: Divide by 2*num_sample here ***
			// if(i==0 && j==0) cout<<"V = "<<V_ij[i][j]<<", ";
			V_ij[i][j] = 1 / (1 + exp(-V_ij[i][j] / 1000));
			// if(i==0 && j==0) cout<<"V' = "<<V_ij[i][j]<<". ";
		}
	}

	// 3.2.4 T_j
	for (int i = 0; i < num_region; i++) // *** Bugfix: Changed num_searcher to num_region ***
		T_j[i] = region_it[i] / region_hl[i];

	// 3.2.5 compute the expected_value
	for (int i = 0; i < num_searcher; i++)
	{
		for (int j = 0; j < num_region; j++)
		{
			expected_value[i][j] = T_j[i] * V_ij[i][j] * M_j[j];
			// if(i==0 && j==0) cout<<"V = "<<V_ij[i][j]<<", ";
			// cout << expected_value[i][j] << "(" << T_j[i] << ", " << V_ij[i][j] << ", " << M_j[j] << ")" << ", ";
			// cout << expected_value[i][j] << ", ";
		}
		// cout << endl;
	}

	// 3.2.6 update sampleV to sample
	for (int i = 0; i < num_searcher; i++)
	{
		for (int j = 0; j < num_region; j++)
		{
			// cout<<"region:"<<j<<endl;
			for (int k = 0; k < num_sample; k++)
			{
				const int m = k << 1;

				if (sampleV_sol_fitness[i][j][m] < sample_sol_fitness[j][k])
				{
					for (int l = 0; l < num_bit_sol; l++)
						sample_sol[j][k][l] = sampleV_sol[i][j][m][l];
					sample_sol_fitness[j][k] = sampleV_sol_fitness[i][j][m];
				}
				if (sampleV_sol_fitness[i][j][m + 1] < sample_sol_fitness[j][k])
				{
					for (int l = 0; l < num_bit_sol; l++)
						sample_sol[j][k][l] = sampleV_sol[i][j][m + 1][l];
					sample_sol_fitness[j][k] = sampleV_sol_fitness[i][j][m + 1];
				}
			}
		}
	}

	//分散區域中的sample
	for (int i = 0; i < num_region; i++)
	{
		for (int j = 1; j < num_sample; j++)
		{

			double same_rate = 0.0;
			i1d tmp_sol;
			tmp_sol.assign(num_bit_sol, -1);
			for (int l = 0; l < num_bit_sol; l++)
			{
				tmp_sol[l] = sample_sol[i][j][l];
				if (tmp_sol[l] == sample_sol[i][0][l])
					same_rate++;
			}

			same_rate /= num_bit_sol;
			// cout<<"same_rate:"<<same_rate<<", ";
			if (same_rate > 0.8)
			{
				for (int l = 0; l < num_bit_sol / 2; l++)
				{
					int rand_num1 = rand() % (num_bit_sol - 1) + 1;
					int rand_num2 = rand() % (num_bit_sol - 1) + 1;
					while (rand_num1 == rand_num2)
						rand_num2 = rand() % (num_bit_sol - 1) + 1;

					int tmp = tmp_sol[rand_num1];
					tmp_sol[rand_num1] = tmp_sol[rand_num2];
					tmp_sol[rand_num2] = tmp;
				}

				double tmp_fitness = evaluate_fitness(tmp_sol);
				if (tmp_fitness < sample_sol_fitness[i][j])
				{
					sample_sol[i][j] = tmp_sol;
					sample_sol_fitness[i][j] = tmp_fitness;
				}
			}
		}
	}
}

// subfunction: 3.2.1 fitness value

double evaluate_fitness(const i1d &sol)
{
	double value = 0.0; // MaxSpan , when cost_time == 't'
	double cost = 0.0;	// Cost	  , when cost_time == 'c'

	vector<int> task_count;
	for (int i = 0; i < num_task.size(); i++)
		task_count.push_back(0);

	// assign tasks to the resource
	Task task_tmp;
	for (int i = 0; i < sol.size(); i++)
	{
		int ii = sol[i];
		int jj = task_count[sol[i]];

		all_task[ii][jj].num_resource = i % num_machine;
		resource[i % num_machine].WorkSet[i / num_machine] = all_task[ii][jj];
		task_count[sol[i]]++;
	}

	// calculate execute time
	for (int i = 0; i < num_bit_sol / num_machine + 1; i++)
	{
		for (int j = 0; j < num_machine; j++)
		{

			if (i >= resource[j].WorkSet.size())
				break;

			Task task_tmp = resource[j].WorkSet[i];

			if (task_tmp.j == 0)
				task_tmp.start = resource[j].execute_time;
			else if (all_task[task_tmp.i][task_tmp.j - 1].num_resource != task_tmp.num_resource &&
					 all_task[task_tmp.i][task_tmp.j - 1].end > resource[j].execute_time)
			{
				task_tmp.start = all_task[task_tmp.i][task_tmp.j - 1].end;
				cost += TDataSize[task_tmp.i][task_tmp.j] * Ctrans[j][all_task[task_tmp.i][task_tmp.j - 1].num_resource];
			}
			else
				task_tmp.start = resource[j].execute_time;

			task_tmp.end = task_tmp.start + Texe[task_tmp.i][j][task_tmp.j];
			resource[j].execute_time = task_tmp.end;
			all_task[task_tmp.i][task_tmp.j] = task_tmp;
			resource[j].WorkSet[i] = task_tmp;
		}

		// calculate Data transfer time
		for (int j = 0; j < num_machine; j++)
		{

			if (i >= resource[j].WorkSet.size() - 1)
				break; // is the last task of this WorkSet

			Task task_tmp = resource[j].WorkSet[i + 1];

			if (task_tmp.j == 0)
				break;
			else if (all_task[task_tmp.i][task_tmp.j - 1].num_resource == j)
				break;
			else
			{
				int last_num_resource = all_task[task_tmp.i][task_tmp.j - 1].num_resource;
				Resource last_resource = resource[last_num_resource];

				double Trans_time = TDataSize[task_tmp.i][task_tmp.j - 1] / Ttrans[last_num_resource][j];

				if (last_resource.execute_time > resource[j].execute_time)
				{
					last_resource.execute_time += Trans_time;
					resource[j].execute_time = last_resource.execute_time;
				}
				else
				{
					resource[j].execute_time += Trans_time;
					last_resource.execute_time = resource[j].execute_time;
				}

				resource[last_num_resource] = last_resource;
			}
		}
	}

	for (int i = 0; i < num_machine; i++)
		if (resource[i].execute_time > value)
			value = resource[i].execute_time;

	for (int i = 0; i < num_machine; i++)
		cost += resource[i].execute_time * Crent[i];

	if (value > 400)
		cost += value * 10;

	count_evaluation++;

	reset_data();

	if (cost_time == 't')
		return cost + value;
	// return cost;
}

// 3.3 select region to which a searcher will be assigned
void vision_selection(int player)
{
	for (int i = 0; i < num_region; i++)
		region_hl[i]++;
	for (int i = 0; i < num_searcher; i++)
	{
		// find the idx of the best vij
		int play0_idx = rand() % num_region;
		double play0_ev = expected_value[i][play0_idx];

		for (int j = 0; j < num_player - 1; j++)
		{
			int play1_idx = rand() % num_region;
			if (expected_value[i][play1_idx] < play0_ev)
			{
				play0_idx = play1_idx;
				play0_ev = expected_value[i][play0_idx];
			}
		}

		// update searcher_sol
		// int j = rand() % num_sample;
		double last_fitness = searcher_sol_fitness[i];
		if (searcher_repeat[i] < 5)
		{
			for (int j = 0; j < num_sample; j++)
			{
				if (sample_sol_fitness[play0_idx][j] < searcher_sol_fitness[i])
				{
					searcher_sol[i] = sample_sol[play0_idx][j];
					searcher_sol_fitness[i] = sample_sol_fitness[play0_idx][j];
				}
			}
		}
		else
		{
			// cout<<"repeat"<<endl;
			searcher_sol[i] = rand_sol();
			searcher_repeat[i] = -1;
		}

		if (searcher_sol_fitness[i] == last_fitness)
			searcher_repeat[i]++;
		else
			searcher_repeat[i] = -1;
		// update region_it[i] and region_hl[i];
		region_it[play0_idx]++;
		region_hl[play0_idx] = 1;
	}

	// similarity for searcher
	for (int i = 1; i < num_searcher; i++)
	{

		double same_rate = 0.0;
		i1d tmp_sol;
		tmp_sol.assign(num_bit_sol, -1);
		for (int l = 0; l < num_bit_sol; l++)
		{
			tmp_sol[l] = searcher_sol[i][l];
			if (tmp_sol[l] == searcher_sol[0][l])
				same_rate++;
		}

		same_rate /= num_bit_sol;
		// cout<<"same_rate:"<<same_rate<<", ";
		if (same_rate > 0.8)
		{
			// cout<<"in"<<endl;
			for (int l = 0; l < num_bit_sol / 2; l++)
			{
				int rand_num1 = rand() % (num_bit_sol - 1) + 1;
				int rand_num2 = rand() % (num_bit_sol - 1) + 1;
				while (rand_num1 == rand_num2)
					rand_num2 = rand() % (num_bit_sol - 1) + 1;

				int tmp = tmp_sol[rand_num1];
				tmp_sol[rand_num1] = tmp_sol[rand_num2];
				tmp_sol[rand_num2] = tmp;
			}

			double tmp_fitness = evaluate_fitness(tmp_sol);
			// if(tmp_fitness < sample_sol_fitness[i][j]){
			searcher_sol[i] = tmp_sol;
			searcher_sol_fitness[i] = tmp_fitness;
			//}
		}
	}

} /*@\label{se:vs:end}@*/

// 4. marketing survey
void marketing_survey() /*@\label{se:ms:begin}@*/
{
	for (int i = 0; i < num_region; i++)
		// 4.1 update region_it
		if (region_hl[i] > 1)
			region_it[i] = 1.0;

	// 4.2 update the best solution
	int j = -1;
	for (int i = 0; i < num_searcher; i++)
	{
		if (searcher_sol_fitness[i] < best_obj_value)
		{
			best_obj_value = searcher_sol_fitness[i];
			j = i;
		}
	}
	if (j >= 0)
		best_sol = searcher_sol[j];

	time_start[2] = clock();
	best_obj_value = two_opt(best_sol, best_obj_value);
	time_end[2] = clock();
	part_time[2] += double(time_end[2] - time_start[2]) / CLOCKS_PER_SEC;
	if (overall_best_obj_value > best_obj_value)
	{
		overall_best_obj_value = best_obj_value;
		overall_best_sol = best_sol;
	}
} /*@\label{se:ms:end}@*/

void read_dataset()
{
	int status = 0; // 0.null 1.Crent 2.Ctrans 3.TDataSize 4.Texe

	//暫存空間
	string line;
	stringstream ssline;
	double ftmp;
	d1d vftmp;
	char *pEnd;
	char *line_tmp;

	while (getline(fin, line))
	{
		if (line[line.size() - 1] == '\r')
			line.resize(line.size() - 1);

		stringstream ssline(line);

		if (line == "C rent")
			status = 1;
		else if (line == "C trans")
			status = 2;
		else if (line == "T trans")
			status = 3;
		else if (line == "T datasize")
			status = 4;
		else if (line == "exe")
			status = 5;
		else if (line == "Fin")
		{
			if (status == 4 || status == 5)
				Texe.resize(Texe.size() + 1);
			status = 0;
		}

		if (status == 1)
		{
			ftmp = strtod(line.c_str(), &pEnd);
			Crent.push_back(ftmp);
		}
		else if (status == 2 || status == 3 || status == 4 || status == 5)
		{
			vftmp.clear();
			while (getline(ssline, line, ','))
			{
				ftmp = strtod(line.c_str(), &pEnd);
				vftmp.push_back(ftmp);
			}
			if (status == 2)
				Ctrans.push_back(vftmp);
			if (status == 3)
				Ttrans.push_back(vftmp);
			if (status == 4)
				TDataSize.push_back(vftmp);
			if (status == 5)
				Texe[Texe.size() - 1].push_back(vftmp);
		}
	}

	Crent.erase(Crent.begin()); //刪除第一個
	Ctrans.erase(Ctrans.begin());
	Ttrans.erase(Ttrans.begin());
	TDataSize.erase(TDataSize.begin());
	Texe.erase(Texe.end() - 1);
	for (int k = 0; k < Texe.size(); k++)
		Texe[k].erase(Texe[k].begin());

	num_machine = Crent.size(); //機器的數量
	for (int i = 0; i < TDataSize.size(); i++)
		num_task.push_back(TDataSize[i].size()); //每個任務的子任務數量

	num_bit_sol = 0;
	for (int i = 0; i < num_task.size(); i++)
		num_bit_sol += num_task[i];
}

i1d rand_sol()
{
	i1d sol;
	for (int i = 0; i < num_task.size(); i++)
		for (int j = 0; j < num_task[i]; j++)
			sol.push_back(i);

	random_shuffle(sol.begin(), sol.end());

	return sol;
}

void reset_data()
{
	// clear

	for (int i = 0; i < resource.size(); i++)
		resource[i].execute_time = 0.0;

	for (int i = 0; i < TDataSize.size(); i++)
	{
		all_task[i].resize(TDataSize[i].size());
		for (int j = 0; j < all_task[i].size(); j++)
		{
			all_task[i][j].i = i;
			all_task[i][j].j = j;
			all_task[i][j].start = 0.0;
			all_task[i][j].end = 0.0;
		}
	}
}

double two_opt(i1d &sol, double best_result)
{
	int limit = 0;
	i1d temp_sol;
	temp_sol.assign(num_bit_sol, 0);
	// cout<<"before 2-opt:"<<best_result;
	int cut1 = rand() % (num_bit_sol - 2) + 1;
	while (limit < 200)
	{
		int cut2 = cut1 + limit;
		while (cut2 >= num_bit_sol)
			cut2 -= num_bit_sol;
		if (cut2 == 0)
			cut2++;

		for (int i = 0; i < num_bit_sol; i++)
		{
			if (i == cut1)
				temp_sol[i] = sol[cut2];
			else if (i == cut2)
				temp_sol[i] = sol[cut1];
			else
				temp_sol[i] = sol[i];
		}

		double temp_result = evaluate_fitness(temp_sol);
		two_opt_eval += 1;
		if (temp_result < best_result)
		{
			for (int i = 0; i < num_bit_sol; i++)
				sol[i] = temp_sol[i];
			best_result = temp_result;
			// limit = 0;
		}

		limit++;
	}
	// cout<<", after 2-opt:"<<best_result<<endl;
	return best_result;
}

void check_sol(const i1d &sol)
{
	i1d coun;
	coun.assign(num_task.size(), 0);
	for (int i = 0; i < num_bit_sol; i++)
		coun[sol[i]]++;
	for (int i = 0; i < coun.size(); i++)
		cout << coun[i] << " ";

	cout << endl;
}
