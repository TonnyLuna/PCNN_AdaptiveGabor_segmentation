/*******************************************************************************
*   AUTHOR: JESUS ANTONIO LUNA ALVAREZ                                         *
*   PROGRAM: PULSE COUPLED NN IMAGE SEGMENTATION (SECUENCIAL)                  *
*   DATE: 02/12/2019                                                           *
*                                                                              *
********************************************************************************
*   COMPILATION:                                                               *
*   gcc segmentation.c -lm -std=c99 -o run                                     *
*                                                                              *
*   RUN:                                                                       *
*   ./run [image path]                                                         *
*                                                                              *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "ppm_image.c"


/*******************************************************************************
*                                 PARAMETERS                                   *
*******************************************************************************/
PPMImage *img;
int r;
int c;
int w = 7;
int h = 7;

float beta = 2.0;
float dT = 1.0;
float Vt = 400.0;

/*******************************************************************************
*                                 MAPS                                         *
*******************************************************************************/
//PPMImage *img; // save image

float W[7][7] = {{0.0000,    0.0002,    0.0011,    0.0018,    0.0011,    0.0002,    0.0000},
                {0.0002,    0.0029,    0.0131,    0.0216,    0.0131,    0.0029,    0.0002},
                {0.0011,    0.0131,    0.0586,    0.0966,    0.0586,    0.0131,    0.0011},
                {0.0018,    0.0216,    0.0966,    0.1592,    0.0966,    0.0216,    0.0018},
                {0.0011,    0.0131,    0.0586,    0.0966,    0.0586,    0.0131,    0.0011},
                {0.0002,    0.0029,    0.0131,    0.0216,    0.0131,    0.0029,    0.0002},
                {0.0000,    0.0002,    0.0011,    0.0018,    0.0011,    0.0002,    0.0000}};

float **F; // input
float **Y; // r x c = 0
float **Th; // r x c = 255
float **T; // output
float **Q;
float **U;


void init(){
  // allocation
  F = (float **) malloc(sizeof(float *) * r);
  Y = (float **) malloc(sizeof(float *) * r);
  T = (float **) malloc(sizeof(float *) * r);
  Th = (float **) malloc(sizeof(float *) * r);
  Q = (float **) malloc(sizeof(float *) * r);
  U = (float **) malloc(sizeof(float *) * r);
  for(int k=0;k<r;k++){
    *(F+k) = (float *) malloc(sizeof(float) * c);
    *(Y+k) = (float *) malloc(sizeof(float) * c);
    *(T+k) = (float *) malloc(sizeof(float) * c);
    *(Th+k) = (float *) malloc(sizeof(float) * c);
    *(Q+k) = (float *) malloc(sizeof(float) * c);
    *(U+k) = (float *) malloc(sizeof(float) * c);
  }

  // Init default values
  for(int i=0;i<r;i++)
    for(int j=0;j<c;j++){
      *(*(Y+i)+j) = 0.0;
      *(*(T+i)+j) = 0.0;
      *(*(Th+i)+j) = 255.0;
    }

  // init image
  int k = 0;
  for(int i=0;i<r;i++)
    for(int j=0;j<c;j++){
      *(*(F+i)+j) = img->gray[k];
      k++;
    }

}

// return max value
int max(int x, int y){
  if(x>y) return x;
  else return y;
}

// return min value
int min(int x, int y){
  if(x<y) return x;
  else return y;
}

// symmetric convolution
float** conv(){
  // allocation
  float **t = (float **) malloc(sizeof(float *) * r);
  for(int k=0;k<r;k++)
    *(t+k) = (float *) malloc(sizeof(float) * c);
  // symmetric
  for(int i=0;i<r;i++)
    for(int j=0;j<c;j++){
      float cm = 0;
      for(int m=-(w/2);m<=(w/2);m++)
        for(int n=-(h/2);n<=(h/2);n++)
          cm += *(*(Y+min(max(i+m, 0), r-1))+min(max(j+n, 0), c-1)) * *(*(W+(m+w/2))+(n+h/2));
      *(*(t+i)+j) = cm;
    }
  return t;
}


int main(int argc, char *argv[]){
  char path[100];
  // read parameters
  if(argc<1){
    printf("Input not found...\n");
    exit(1);
  } else
    strcpy(path, argv[1]);

  img = readPPM(path);
  grayScale(img);
  r = img->x;
  c = img->y;
  printf("%d %d\n", r, c);

  // allocation and init
  init();
  // img save
  saveGray("input.ppm", F, r, c);

  // init time count
  struct timeval stop,start;
  gettimeofday(&start,NULL);
  printf("=====================\n");
  printf("Start\n");

  int fire_num = 0;
  int n = 0;
  while(fire_num < r*c){
    n++;
    float **L = conv();
    //Th = Th - dT + Vt*Y;
    for(int i=0;i<r;i++)
      for(int j=0;j<c;j++)
        *(*(Th+i)+j) = *(*(Th+i)+j) - dT + Vt * *(*(Y+i)+j);
    int fire = 1;
    while(fire == 1){
      // equal counter
      int eq = 1;
      for(int i=0;i<r;i++)
        for(int j=0;j<c;j++){
          // Q = Y;
          *(*(Q+i)+j) = *(*(Y+i)+j);
          // U = F.*(1 + beta*L);
          *(*(U+i)+j) = *(*(F+i)+j) * (1.0 + beta * *(*(L+i)+j));
          //Y = double(U > Th);
          *(*(Y+i)+j) = (float) (*(*(U+i)+j) > *(*(Th+i)+j));
          // isequal(Q,Y)
          if(*(*(Q+i)+j) != *(*(Y+i)+j))
            eq = 0;
        }
      if(eq)
        fire = 0;
      else
        L = conv();
    }
    int k = 0;
    for(int i=0;i<r;i++)
      for(int j=0;j<c;j++){
        //T = T + n.*Y;
        *(*(T+i)+j) = *(*(T+i)+j) + (n * *(*(Y+i)+j));
        // sum (Y)
        k += *(*(Y+i)+j);
      }

    fire_num = fire_num + k;
    printf("Completed: %.1f %c\r",  ((float)fire_num / (float)(r*c)) * 100.0, '%');
    fflush(stdout);
  }
  // T = 256 - T;
  for(int i=0;i<r;i++)
    for(int j=0;j<c;j++)
      *(*(T+i)+j) = 256.0 - *(*(T+i)+j);

  // end time count
  gettimeofday(&stop,NULL);
  printf("=====================\n");
  printf("time: %f sec\n",(double) (stop.tv_usec - start.tv_usec) / 1000000 +
         (double) (stop.tv_sec - start.tv_sec));

  // img save
  saveGray("output.ppm", T, r, c);

  return 0;
}




//
