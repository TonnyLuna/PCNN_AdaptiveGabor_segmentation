/*******************************************************************************
*   AUTHOR: JESUS ANTONIO LUNA ALVAREZ                                         *
*   PROGRAM: PULSE COUPLED NN IMAGE SEGMENTATION (GPU)                         *
*   DATE: 04/12/2019                                                           *
*                                                                              *
********************************************************************************
*   COMPILATION:                                                               *
*   nvcc segmentation.cu -o run                                                *
*                                                                              *
*   RUN:                                                                       *
*   ./run [image path]                                                         *
*                                                                              *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <string.h>
#include "imageIO.c"


/*******************************************************************************
*                                 PARAMETERS                                   *
*******************************************************************************/
Image *img;
int r;
int c;
int kernelSize = 5; // mas de 5 sigue siendo lo mismo

float beta = 2.0;
float dT = 1.0;
float Vt = 400.0;

/*******************************************************************************
*                                 MAPS                                         *
*******************************************************************************/

float *F;
float *L;
float *T;
float *Y;
float *Q;
float *U;
float *W;
float *G;
float *Th;
float *NE;
float *T_temp;
int *fr;

__host__ float* init(){
  // allocation
  float *F = (float *) malloc(sizeof(float) * r * c);
  // init image
  for(int i=0;i<r*c;i++)
    *(F+i) = img->gray[i];
  return F;
}


__host__ float* gaussian_filter(int size, float sigma){
  float **W = (float **) malloc(sizeof(float *) * size);
  for(int i=0;i<size;i++)
    *(W+i) = (float *) malloc(sizeof(float) * size);

  float r;
  float s = 2.0 * sigma * sigma;
  float sum = 0.0;

  for (int x = -(size/2); x <= (size/2); x++) {
    for (int y = -(size/2); y <= (size/2); y++) {
      r = sqrt(x * x + y * y);
      *(*(W+ x + (size/2))+ y + (size/2)) = (exp(-(r * r) / s)) / (M_PI * s);
      sum += W[x + (size/2)][y + (size/2)];
    }
  }

  float *w = (float *) malloc(sizeof(float *) * size * size);
  for (int i = 0; i < size;i++)
    for (int j = 0; j < size;j++)
      *(w + size*i + j) = *(*(W+i)+j) / sum;
  return w;
}


__host__ float* gabor_filter(int size, float sigma, float theta, float lambda, float psi, float gamma){
  float *w = (float *) malloc(sizeof(float *) * size * size);
  int k=0;
  for (int x = -(size/2); x <= (size/2); x++)
    for (int y = -(size/2); y <= (size/2); y++){
      float xz = x * cos(theta) + y * sin(theta);
      float yz = -x * sin(theta) + y * cos(theta);
      *(w+k) = exp(-(xz*xz + gamma*gamma*yz*yz)/(2*sigma*sigma)) * cos(2*M_PI*(xz/lambda)+psi);
      k++;
    }
  return w;
}


__host__ float* noise_estimation_filter(int size){
  float* ne = (float*)malloc(sizeof(float)*size*size);
  *(ne+0) = 1.0;*(ne+1) = -2.0;*(ne+2) = 1.0;
  *(ne+3) = -2.0;*(ne+4) = 4.0;*(ne+5) = -2.0;
  *(ne+6) = 1.0;*(ne+7) = -2.0;*(ne+8) = 1.0;
  return ne;
}


__host__ float** to2D(float *x){
    // allocation
  float **y = (float **) malloc(sizeof(float *) * r);
  for(int i=0;i<r;i++)
    *(y+i) = (float *) malloc(sizeof(float) * c);
  // init image
  for(int i=0;i<r*c;i++)
    *(*(y+(int)(i/c))+(int)(i%c)) = *(x+i);
  return y;
}


__global__ void initKernel(float *Y, float *T, float *Th, float *T_temp){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  *(Th+i) = 255.0;
  *(Y+i) = 0.0;
  *(T+i) = 0.0;
  *(T_temp+i) = 0.0;
}


__global__ void conv(float *L, float *Y, float *W, int r, int c, int size){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float cm = 0.0;
  for(int m=-(size/2);m<=(size/2);m++){
    for(int n=-(size/2);n<=(size/2);n++){
	    int k = (min(max((i/r) +m, 0), c-1) * r + min(max((int)(i%r)+n, 0), r-1));
      cm += *(Y+k) * *(W + size * (m+size/2) + (n+size/2));
	  }
  }
  *(L+i) = cm;
}


__host__ float noise_estimation(float *O, float *I, float *NE, int r, int c){
    conv<<<r,c>>>(O, I, NE, r, c, 3);
    float *o = (float *) malloc(sizeof(float) * r * c);
    cudaMemcpy(o, O, sizeof(float) * r * c, cudaMemcpyDeviceToHost);

    float sigma = 0.0;
    for(int i=0;i<r*c;i++)
      sigma += abs(*(o+i));

    return sigma*sqrt(0.5*M_PI)/(6.0*(r-2)*(c-2));
}


__global__ void dilation(float *O, float *I, int r, int c, int size){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float max_v = 0.0;
  int first = -(size/2);
  if(size %2==0) first++;
  int last = size/2;
  for(int m=first;m<=last;m++)
    for(int n=first;n<=last;n++){
	    int k = (min(max((i/r) +m, 0), r-1) * r + min(max((int)(i%r)+n, 0), r-1));
      if(*(I+k) > max_v){
        max_v = *(I+k);
        m = -(size/2);
        n = -(size/2);
      }
      *(O+k) = max_v;
	  }
}


__global__ void erosion(float *O, float *I, int r, int c, int size){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float min_v = 256.0;
  int first = -(size/2);
  if(size %2==0) first++;
  int last = size/2;
  for(int m=first;m<=last;m++)
    for(int n=first;n<=last;n++){
	    int k = (min(max((i/r) +m, 0), r-1) * r + min(max((int)(i%r)+n, 0), r-1));
      if(*(I+k) < min_v){
        min_v = *(I+k);
        m = -(size/2);
        n = -(size/2);
      }
      *(O+k) = min_v;
	  }
}


__global__ void median_filter(float *L, float *Y, int r, int c, int size){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float kernel[7*7];

  int l=0;
  for(int m=-(size/2);m<=(size/2);m++)
    for(int n=-(size/2);n<=(size/2);n++){
	    int k = (min(max((i/r) +m, 0), r-1) * r + min(max((int)(i%r)+n, 0), r-1));
      kernel[l++] = *(Y+k);
	  }

  for(int m=1;m<size;m++)
    for(int n=0;n<size-m;n++)
      if(kernel[n]>kernel[n+1]){
        const float aux = kernel[n+1];
        kernel[n+1] = kernel[n];
        kernel[n] = aux;
      }

  *(L+i) = kernel[size/2];
}


__global__ void mean_filter(float *L, float *Y, int r, int c, int size){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float mean = 0.0;
  for(int m=-(size/2);m<=(size/2);m++)
    for(int n=-(size/2);n<=(size/2);n++){
	    int k = (min(max((i/r) +m, 0), r-1) * r + min(max((int)(i%r)+n, 0), r-1));
      mean += *(Y+k);
	  }

  *(L+i) = mean/(size*size);
}


__host__ void histogram(float *t, int *fr, int r, int c){
  for(int i=0;i<r*c;i++){
    int ind = (int) *(t+i);
    *(fr+ind) += 1;
  }
}


__host__ float treshold(int *fr){
  int f = 0;
  int ind = 0;
  for(int i=0;i<255;i++){
    if(*(fr+i) > f){
      f = *(fr+i);
      ind = i;
    }
  }
  return (float)ind;
}

//Th = Th - dT + Vt*Y;
__global__ void acum(float *Th, float *Y, float dT, float Vt){
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   *(Th+i) = *(Th+i) - dT + Vt * *(Y+i);
}

//T = T + n.*Y;
__global__ void Tacum(float *T, float *Y, float n){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  *(T+i) = *(T+i) + n * *(Y+i);
}

//K = sum(Y);
__global__ void Kacum(float *Y, int *K){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(K, (int)*(Y+i));
}

//T = T + n.*Y;
__global__ void output(float *T){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  *(T+i) = 256.0 - *(T+i);
}


__global__ void inverse(float *T){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  *(T+i) = 255.0 - *(T+i);
}


__global__ void binarize(float *T, float th){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(*(T+i) != th)
    *(T+i) = 0.0;
  else
    *(T+i) = 255.0;
}


__global__ void pulses(float *F, float *Y, float *Q, float *U, float *L, float *Th, int *EQ, float beta){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Q = Y
  *(Q+i) = *(Y+i);
  // U = F.*(1 + beta*L);
  *(U+i) = *(F+i) * (1.0 + beta * *(L+i));
  //Y = double(U > Th);
  *(Y+i) = (float) (*(U+i) > *(Th+i));
  // isequal(Q,Y)
  if(*(Q+i) != *(Y+i))
    *EQ = 1;
}


__global__ void set(int *x, int value){
  *x = value;
}


int main(int argc, char *argv[]){
  char path[100];
  char outpath[100];
  // read parameters
  if(argc<2){
    printf("Input not found...\n");
    exit(1);
  } else {
    strcpy(path, argv[1]);
    strcpy(outpath, argv[2]);
  }
  img = read(path);
  grayScale(img);
  stretching(img);
  //inverse(img);

  r = img->x;
  c = img->y;
  printf("R x C: %d %d\n", r, c);

  // allocation and init
  float *f = init();
  // img save
  //saveGray("input.ppm", to2D(f), r, c);

  fr = (int*)malloc(sizeof(int)*256);
  for(int i=0;i<256;i++)
    *(fr+i) = 0;
  cudaMalloc( (void**)&F, sizeof(float)*r*c);
  cudaMemcpy(F, f, sizeof(float) * r * c, cudaMemcpyHostToDevice);

  cudaMalloc( (void**)&Th, sizeof(float)*r*c);
  cudaMalloc( (void**)&Y, sizeof(float)*r*c);
  cudaMalloc( (void**)&Q, sizeof(float)*r*c);
  cudaMalloc( (void**)&U, sizeof(float)*r*c);
  cudaMalloc( (void**)&L, sizeof(float)*r*c);
  cudaMalloc( (void**)&T, sizeof(float)*r*c);
  cudaMalloc( (void**)&T_temp, sizeof(float)*r*c);

  float *ne = noise_estimation_filter(3);
  cudaMalloc( (void**)&NE, sizeof(float)*3*3);
  cudaMemcpy(NE, ne, sizeof(float) * 3*3, cudaMemcpyHostToDevice);
  float noise = noise_estimation(Y, F, NE, r, c);
  //printf("Noise: %f\n", noise);
  if(noise<5.0)noise+=5.0;

  float *w = gabor_filter(kernelSize, 1.0, 0.0, noise, 0.0, 1.0);
  cudaMalloc( (void**)&W, sizeof(float)*kernelSize*kernelSize);
  cudaMemcpy(W, w, sizeof(float) * kernelSize*kernelSize, cudaMemcpyHostToDevice);

  initKernel<<<r,c>>>(Y, T, Th, T_temp);

  int *K;
  int *EQ;
  int *k = (int*)malloc(sizeof(int));
  int *eq = (int*)malloc(sizeof(int));
  cudaMalloc( (void**)&K, sizeof(int));
  cudaMalloc( (void**)&EQ, sizeof(int));

  // init time count
  struct timeval stop,start;
  gettimeofday(&start,NULL);
  printf("=====================\n");
  printf("Start\n");

  int n = 0;
  int fire_num = 0;
  while(fire_num < r*c){
    n++;
  	conv<<<r,c>>>(L, Y, W, r, c, kernelSize);
    acum<<<r,c>>>(Th, Y, dT, Vt);
    int fire = 1;
    while(fire == 1){
      set<<<1,1>>>(EQ, 0);
      pulses<<<r,c>>>(F, Y, Q, U, L, Th, EQ, beta);
  	  cudaMemcpy(eq, EQ, sizeof(int), cudaMemcpyDeviceToHost);
  	  if(*eq!=1)
        fire = 0;
      else
        conv<<<r,c>>>(L, Y, W, r, c, kernelSize);
  	}
    set<<<1,1>>>(K, 0);
  	Tacum<<<r,c>>>(T, Y, n);
    Kacum<<<r,c>>>(Y, K);
  	cudaMemcpy(k, K, sizeof(int), cudaMemcpyDeviceToHost);
    fire_num += *k;
  	printf("Completed: %.1f %c\r",  ((float)fire_num / (float)(r*c)) * 100.0, '%');
    fflush(stdout);
  }
  output<<<r,c>>>(T);
  float *t = (float *) malloc(sizeof(float) * r * c);
  cudaMemcpy(t, T, sizeof(float) * r * c, cudaMemcpyDeviceToHost);
  histogram(t, fr, r, c);
  binarize<<<r,c>>>(T, treshold(fr));

  cudaMemcpy(t, T, sizeof(float) * r * c, cudaMemcpyDeviceToHost);

  // end time count
  gettimeofday(&stop,NULL);
  printf("=====================\n");
  printf("iterations: %d\n", n);
  printf("time: %f sec\n",(double) (stop.tv_usec - start.tv_usec) / 1000000 +
         (double) (stop.tv_sec - start.tv_sec));
  // img save
  saveGray(outpath, to2D(t), r, c);
  //saveBinarized("output.ppm", to2D(t), r, c, -1);

  return 0;
}
