#include<iostream>
#include<bits/stdc++.h>

using namespace std;

int main(){
    FILE *f1 = fopen("/home/vision/Desktop/code_c_c++/my_BA/original_data/problem-16-22106-pre.txt", "r");
    FILE *f2 = fopen("/home/vision/Desktop/code_c_c++/my_BA/raw_data/mini_data.txt", "w");
    //generate raw/mini_data_2.txt  select_n_camera = 16, select_n_point = 6, select_n_observation = 52
    //generate raw/mini_data.txt  select_n_camera = 16, select_n_point = 1000, select_n_observation = 8037
    //generate raw/big_data.txt  select_n_camera = n_camera, select_n_point = n_point, select_n_observation = n_observation

    int n_camera, n_point, n_observation;
    fscanf(f1, "%d", &n_camera);
    fscanf(f1, "%d", &n_point);
    fscanf(f1, "%d", &n_observation);

    int select_n_camera=16, select_n_point=1000, select_n_observation=8037; //选择提取的camera个数，观测点的个数，观测数量（数一数）
    fprintf(f2, "%d %d %d\n", select_n_camera, select_n_point, select_n_observation);
    
    for (int i = 0; i < n_observation;i++){
        int c_id, p_id;
        double u, v;
        fscanf(f1, "%d%d%lf%lf", &c_id, &p_id, &u, &v);
        if(i < select_n_observation){
            fprintf(f2, "%d %d %.6e %.6e\n", c_id, p_id, u, v);
        }       
    }

    for (int i = 0; i < n_camera;i++){
        for (int j = 0; j < 9;j++){
            double c;
            fscanf(f1, "%lf", &c);
            if(i < select_n_camera){
                fprintf(f2, "%.16e\n", c);
            }
        }
    }

    for (int i = 0; i < select_n_point;i++){
        for (int j = 0; j < 3;j++){
            double p;
            fscanf(f1, "%lf", &p);
            fprintf(f2, "%.16e\n", p);
        }
    }

    fclose(f1);
    fclose(f2);

    return 0;
}