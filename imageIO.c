#include<stdio.h>
#include<stdlib.h>
//#include<png.h>

typedef struct{
  unsigned char red, green, blue;
}Pixel;

typedef struct{
  int x, y;
  float *gray; // [0, 255]
  Pixel *data;
}Image;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

float gray_max(Image *img){
  float max = -9999.9;
  for(int i=0;i<img->x*img->y;i++)
    if(img->gray[i] > max)
      max = img->gray[i];
  return max;
}


float gray_min(Image *img){
  float min = 9999.9;
  for(int i=0;i<img->x*img->y;i++)
    if(img->gray[i] < min)
      min = img->gray[i];
  return min;
}


Image* readPPM(const char *filename){
  char buff[16];
  Image *img;
  FILE *fp;
  int c, rgb_comp_color;
  //open PPM file for reading
  fp = fopen(filename, "rb");
  if (!fp){
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  //read image format
  if (!fgets(buff, sizeof(buff), fp)){
    perror(filename);
    exit(1);
  }

  //check the image format
  if (buff[0] != 'P' || buff[1] != '6'){
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  //alloc memory form image
  img = (Image *)malloc(sizeof(Image));
  if (!img){
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  //check for comments
  c = getc(fp);
  while (c == '#'){
    while (getc(fp) != '\n');
      c = getc(fp);
  }

  ungetc(c, fp);
  //read image size information
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2){
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  //read rgb component
  if (fscanf(fp, "%d", &rgb_comp_color) != 1){
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  //check rgb component depth
  if (rgb_comp_color!= RGB_COMPONENT_COLOR){
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n') ;
  //memory allocation for pixel data
  img->data = (Pixel*)malloc(img->x * img->y * sizeof(Pixel));

  if (!img){
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  //read pixel data from file
  if (fread(img->data, 3 * img->x, img->y, fp) != img->y){
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}


Image* readBMP(char *filename){
  unsigned char header[54];
  Image *img;
  img = (Image *)malloc(sizeof(Image));

  FILE *fd;
  fd = fopen(filename, "rb");
  if (fd == NULL){
    fprintf(stderr, "Error: fopen failed '%s'\n", filename);
    exit(1);
  }
  // Read header
  fread(header, sizeof(unsigned char), 54, fd);

  // Capture dimensions
  img->x = *(int*)&header[18];
  img->y = *(int*)&header[22];

  img->data = (Pixel*)malloc(img->x * img->y * sizeof(Pixel));
  // Compute new width, which includes padding
  int widthnew = img->x*4;// + padding;
  // Allocate temporary memory to read widthnew size of data
  unsigned char* data = (unsigned char *)malloc(widthnew * sizeof (unsigned int));
  // Read row by row of data and remove padded data.
  int k = img->x * img->y;
  for (int i=0;i<img->y;i++){
    fread(data, sizeof(unsigned char), widthnew, fd);
    // BGR -> RGB format
    for (int j=img->x*4;j>0;j-=4){
      img->data[k].red = data[j + 2];
      img->data[k].green = data[j + 1];
      img->data[k].blue = data[j + 0];
      k--;
    }
  }
  free(data);
  fclose(fd);
  return img;
}


Image* read(char *filename){
  // detect type of image input
  char *type = (char *)malloc(sizeof(char) * 3);
  int t = -1;
  for (int i=0;i<100;i++) {
    if(*(filename+i) == '.')
      t++;
    else
    if(t>=0 && t<4){
      *(type+t) = *(filename+i);
      t++;
    } else if(t>=4)
      break;
  }
  Image *img;
  if(*(type) == 'b' && *(type+1) == 'm' && *(type+2) == 'p')
    img = readBMP(filename);
  else if(*(type) == 'p' && *(type+1) == 'p' && *(type+2) == 'm')
    img = readPPM(filename);

  return img;
}


void writePPM(const char *filename, Image *img){
  FILE *fp;
  //open file for output
  fp = fopen(filename, "wb");
  if (!fp){
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  //write the header file
  //image format
  fprintf(fp, "P6\n");

  //comments
  fprintf(fp, "# Created by %s\n",CREATOR);

  //image size
  fprintf(fp, "%d %d\n",img->x,img->y);

  // rgb component depth
  fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

  // pixel data
  fwrite(img->data, 3 * img->x, img->y, fp);
  fclose(fp);
}


void grayScale(Image *img){
  img->gray = (float*)malloc(img->x * img->y * sizeof(float));
  if(img){
    for(int i=0;i<img->x*img->y;i++){
      int R = img->data[i].red;
      int G = img->data[i].green;
      int B = img->data[i].blue;

      img->data[i].red=(R+G+B)/3;
      img->data[i].green=(R+G+B)/3;
      img->data[i].blue=(R+G+B)/3;

      img->gray[i] = ((R+G+B)/3);
    }
  }
}


void inverse(Image* img){
  if(img){
    for(int i=0;i<img->x*img->y;i++){
      img->data[i].red = 255 - img->data[i].red;
      img->data[i].green = 255 - img->data[i].green;
      img->data[i].blue = 255 - img->data[i].blue;
      img->gray[i] = 255 - img->gray[i];
    }
  }
}


void stretching(Image *img){
  if(img){
    float min = gray_min(img);
    float max = gray_max(img);
    for(int i=0;i<img->x*img->y;i++)
      img->gray[i] = ((img->gray[i] - min)/(max - min)) * 255.0;
  }
}


int max_frecuency(float **I, int w, int h){
  int *f = (int*)malloc(sizeof(int)*w*h);
  for(int i=0;i<w*h;i++)
    *(f+i) = 0;

  for(int i=0;i<w;i++)
    for(int j=0;j<h;j++)
      *(f+(int)*(*(I+i)+j)) += 1;

  int max = -9999.99;
  int max_ind = 0;
  for(int i=0;i<w*h;i++)
    if(*(f+i) > max){
      max = *(f+i);
      max_ind = i;
    }
  return max_ind;
}


void saveBinarized(const char *filename, float **I, int w, int h, int t){
  Image *img;
  img = (Image *)malloc(sizeof(Image));
  img->data = (Pixel*)malloc(w * h * sizeof(Pixel));
  img->x = w;
  img->y = h;

  int k = 0;
  if(t > 0){
    for(int i=0;i<w;i++)
      for(int j=0;j<h;j++){
        if(*(*(I+i)+j) < t){
          img->data[k].red = 0;
          img->data[k].green = 0;
          img->data[k].blue = 0;
        }
        else{
          img->data[k].red = 255;
          img->data[k].green = 255;
          img->data[k].blue = 255;
        }
        k++;
      }
  }
  else if(t == -1){
    t = max_frecuency(I, w, h);
    for(int i=0;i<w;i++)
      for(int j=0;j<h;j++){
        if(*(*(I+i)+j) != t){
          img->data[k].red = 0;
          img->data[k].green = 0;
          img->data[k].blue = 0;
        }
        else{
          img->data[k].red = 255;
          img->data[k].green = 255;
          img->data[k].blue = 255;
        }
        k++;
      }
  }
  else{
    for(int i=0;i<w;i++)
      for(int j=0;j<h;j++){
        img->data[k].red = *(*(I+i)+j);
        img->data[k].green = *(*(I+i)+j);
        img->data[k].blue = *(*(I+i)+j);
        k++;
      }
  }
  writePPM(filename, img);
}


void saveGray(const char *filename, float **I, int w, int h){
  Image *img;
  img = (Image *)malloc(sizeof(Image));
  img->data = (Pixel*)malloc(w * h * sizeof(Pixel));
  img->x = w;
  img->y = h;

  int k = 0;
  for(int i=0;i<w;i++)
    for(int j=0;j<h;j++){
      img->data[k].red = *(*(I+i)+j);
      img->data[k].green = *(*(I+i)+j);
      img->data[k].blue = *(*(I+i)+j);
      k++;
    }

  writePPM(filename, img);
}



//
