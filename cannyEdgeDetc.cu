/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *	     
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
#include "jpeglib.h"

/**
 * IMAGE DATA FORMATS:
 *
 * The standard input image format is a rectangular array of pixels, with
 * each pixel having the same number of "component" values (color channels).
 * Each pixel row is an array of JSAMPLEs (which typically are unsigned chars).
 * If you are working with color data, then the color values for each pixel
 * must be adjacent in the row; for example, R,G,B,R,G,B,R,G,B,... for 24-bit
 * RGB color.
 */

/* The "frame structure" structure contains an image frame (in RGB or grayscale
 * formats) for passing around the CS338 projects.
 */
typedef struct frame_struct {
  JSAMPLE *image_buffer;        /* Points to large array of R,G,B-order/grayscale data
				 * Access directly with:
				 *   image_buffer[num_components*pixel + component]
				 */
  JSAMPLE **row_pointers;       /* Points to an array of pointers to the beginning
				 * of each row in the image buffer.  Use to access
				 * the image buffer in a row-wise fashion, with:
				 *   row_pointers[row][num_components*pixel + component]
				 */
  int image_height;             /* Number of rows in image */
  int image_width;              /* Number of columns in image */
  int num_components;   /* Number of components (usually RGB=3 or gray=1) */
} frame_struct_t;


typedef frame_struct_t *frame_ptr;


#define MAXINPUTS 1
#define MAXOUTPUTS 1

frame_ptr input_frames[MAXINPUTS];      /* Pointers to input frames */
frame_ptr output_frames[MAXOUTPUTS];    /* Pointers to output frames */

/* Read/write JPEGs, for program startup & shutdown */
void write_JPEG_file (char * filename, frame_ptr p_info, int quality);
frame_ptr read_JPEG_file (char * filename);

/* Allocate/deallocate frame buffers, USE AS NECESSARY! */
frame_ptr allocate_frame(int height, int width, int num_components);
void destroy_frame(frame_ptr kill_me);

/**
 * write_JPEG_file writes out the contents of an image buffer to a JPEG.
 * A quality level of 2-100 can be provided (default = 75, high quality = ~95,
 * low quality = ~25, utter pixellation = 2).  Note that unlike read_JPEG_file,
 * it does not do any memory allocation on the buffer passed to it.
 */
void write_JPEG_file (char * filename, frame_ptr p_info, int quality) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE * outfile;               /* target file */

  /* Step 1: allocate and initialize JPEG compression object */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  /* Step 2: specify data destination (eg, a file) */
  /* Note: steps 2 and 3 can be done in either order. */
  if ((outfile = fopen(filename, "wb")) == NULL) {
    fprintf(stderr, "ERROR: Can't open output file %s\n", filename);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outfile);

  /* Step 3: set parameters for compression */

  /* Set basic picture parameters (not optional) */
  cinfo.image_width = p_info->image_width;      /* image width and height, in pixels */
  cinfo.image_height = p_info->image_height;
  cinfo.input_components = p_info->num_components; /* # of color components per pixel */
  if (p_info->num_components == 3)
    cinfo.in_color_space = JCS_RGB;     /* colorspace of input image */
  else if (p_info->num_components == 1)
    cinfo.in_color_space = JCS_GRAYSCALE;
  else {
    fprintf(stderr, "ERROR: Non-standard colorspace for compressing!\n");
    exit(1);
  }
  /* Fill in the defaults for everything else, then override quality */
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

  /* Step 4: Start compressor */
  jpeg_start_compress(&cinfo, TRUE);

  /* Step 5: while (scan lines remain to be written) */
  /*           jpeg_write_scanlines(...); */
  while (cinfo.next_scanline < cinfo.image_height) {
    (void) jpeg_write_scanlines(&cinfo, &(p_info->row_pointers[cinfo.next_scanline]), 1);
  }

  /* Step 6: Finish compression & close output */
  jpeg_finish_compress(&cinfo);
  fclose(outfile);

  /* Step 7: release JPEG compression object */
  jpeg_destroy_compress(&cinfo);
}

/**
 * read_JPEG_file reads the contents of a JPEG into an image buffer, which
 * is automatically allocated after the size of the image is determined.
 * We want to return a frame struct on success, NULL on error.
 */
frame_ptr read_JPEG_file (char * filename) {
  /* This struct contains the JPEG decompression parameters and pointers to
   * working space (which is allocated as needed by the JPEG library).
   */
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE * infile;                /* source file */
  frame_ptr p_info;             /* Output frame information */

  /* Step 1: allocate and initialize JPEG decompression object */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  /* Step 2: open & specify data source (eg, a file) */
  if ((infile = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "ERROR: Can't open input file %s\n", filename);
    exit(1);
  }
  jpeg_stdio_src(&cinfo, infile);

  /* Step 3: read file parameters with jpeg_read_header() */
  (void) jpeg_read_header(&cinfo, TRUE);

  /* Step 4: use default parameters for decompression */

  /* Step 5: Start decompressor */
  (void) jpeg_start_decompress(&cinfo);

  /* Step X: Create a frame struct & buffers and fill in the blanks */
  fprintf(stderr, "  Opened %s: height = %d, width = %d, c = %d\n",
	  filename, cinfo.output_height, cinfo.output_width, cinfo.output_components);
  p_info = allocate_frame(cinfo.output_height, cinfo.output_width, cinfo.output_components);

  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */
  while (cinfo.output_scanline < cinfo.output_height) {
    (void) jpeg_read_scanlines(&cinfo, &(p_info->row_pointers[cinfo.output_scanline]), 1);
  }

  /* Step 7: Finish decompression */
  (void) jpeg_finish_decompress(&cinfo);

  /* Step 8: Release JPEG decompression object & file */
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);

  return p_info;
}

/**
 * allocate/destroy_frame allocate a frame_struct_t and fill in the
 * blanks appropriately (including allocating the actual frames), and
 * then destroy them afterwards.
 */
frame_ptr allocate_frame(int height, int width, int num_components) {
  int row_stride;               /* physical row width in output buffer */
  int i;
  frame_ptr p_info;             /* Output frame information */

  /* JSAMPLEs per row in output buffer */
  row_stride = width * num_components;

  /* Basic struct and information */
  if ((p_info = (frame_struct_t*)malloc(sizeof(frame_struct_t))) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  p_info->image_height = height;
  p_info->image_width = width;
  p_info->num_components = num_components;

  /* Image array and pointers to rows */
  if ((p_info->row_pointers = (JSAMPLE**)malloc(sizeof(JSAMPLE *) * height)) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  if ((p_info->image_buffer = (JSAMPLE*)malloc(sizeof(JSAMPLE) * row_stride * height)) == NULL){
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  for (i=0; i < height; i++)
    p_info->row_pointers[i] = & (p_info->image_buffer[i * row_stride]);

  /* And send it back! */
  return p_info;
}

void destroy_frame(frame_ptr kill_me) {
  free(kill_me->image_buffer);
  free(kill_me->row_pointers);
  free(kill_me);
}

//////////////////////////////////////////////////////////////////////////////////
/////////////////// My Code Starts Here //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

/**
 * For double threshold analysis, these are the two thresholds.
 * Atleast from the input images I've looked on, these have shown good
 * results.
 **/
int low_threshold = 30;
int high_threshold = 70;

/**
 * For Gaussian kernel, this is mainly due to popularity with
 * different people who implemented the algorithm.
 **/
#define SIGMA 1.4


/**
 * the size of a sobel operator: gradient in x and y direction. 
 **/
#define SOBEL_MASK_SIZE 9

/**
 * size of a gaussian mask. I am using a 5x5 kernel mask, with sigma = 1.4
 * and k = 2.
 **/
#define GAUSSIAN_MASK_SIZE 25

/**
 * Global sobel operators-- one in the x direction and the other in the y direction.
 * The two operators are used to find gradient changes (by convolution) in an image.
 * they are also used to find gradient magnitude and the angle of a pixel, which is used
 * for non-maximum suppression to bring out edges.
 **/
int xGradient[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
int yGradient[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

/**
 * runs a kernel with specification as: 
 * no use of constant memory and shared memory.It takes a frame_ptr
 * as the output of the edge detection algorithm. The input is found in
 * input_frames[0].
 **/
void runKernel(frame_ptr result);

/**
 * runs a device kernel with either constant memory, shared memory, or both
 * turned on. To use shared memory and tiling, #define RUN_CONST_SHARED_MEM_KERNEL.
 * if not defined, the default kernel executed is the one that only uses constant memory
 * without tiling.
 **/
void run_const_shared_mem_kernel(frame_ptr result);

/**
 * this runs the sequential version of canny edge detector. No device memory
 * or compute power is used here.
 **/
void runSequential(frame_ptr output);

/**
 * Given a k and sigma, it computes a gaussian filter kernel of size
 * (2k + 1) x (2k + 1)
 **/
void set_convo_kernel(float *kernel, int k, float sigma);

// memcpy error in the definition of the hysteresis analysis using
// shared data.
#define RUN_CONST_SHARED_MEM_KERNEL

/**
 * For testing the output of each step. That way you can get what it looks
 * like to have an xGradient Applied only. whereas in the other implementations,
 * this step is combined with magnitude and angle together.
 * -- To get this, uncomment the define, and set prepareKernelCall() to call the 
 * runKernel() function.
 **/
//#define RUN_INDIVIDUAL_STEPS

/**
 * Makes sure values match in the two images
 * @credits: Professor Kelly.
 **/
void checkResults(frame_ptr f1, frame_ptr f2) {
  int i, j, k;

  if(f1->image_height != f2->image_height && f1->image_width != f2->image_width
     && f1->num_components != f2->num_components){
    fprintf(stderr, "Dimensions do not match\n");
    exit(1);
  }

  for (i=0; i < f1->image_height; i++){
    for (j=0; j < f1->image_width; j++){
      for (k=0; k < f1->num_components; k++){
	JSAMPLE j1 = f1->row_pointers[i][(f1->num_components)*j+k];
	JSAMPLE j2 = f2->row_pointers[i][(f2->num_components)*j+k];
	if(j1 != j2){
	  fprintf(stderr, "Values do not match at (%d, %d, %d) \n", i, j, k);
	  fprintf(stderr, "from %d\n", j1);
	  fprintf(stderr, "to %d\n", j2);
	  exit(1);
	}
      }
    }
  }
}

/**
 * converts a colored image (with R, G, B values) into a grayscale
 * image using the conversion ratios of 
 *    newpixel = R*0.2125 + G*0.7154 + B*0.0721
 * If the image given is in grayscale already, nothing is done.
 *
 * @required src.height == dst.height && src.width == src.height
 * @param src is the colored image
 * @param dst is the output of the conversion
 */
void toGrayScale(frame_ptr src, frame_ptr dst) {
  if (src->num_components == 1) {
    // iterates over the entire image and do a 1-1 copy since
    // input image is already in grayscale.
    for (int i = 0; i < src->image_height; i++) {
      for (int j = 0; j < src->image_width; j++) {
	dst->row_pointers[i][j] = src->row_pointers[i][j];
      }
    } 
  } else {
    // iterates over the entire image and apply the
    // conversion ratios to create a grayscale image.
    for (int i = 0; i < src->image_height; i++) {
      for (int j = 0; j < src->image_width; j++) {
	dst->row_pointers[i][j] = src->row_pointers[i][3*j] * 0.2125
	  + src->row_pointers[i][3*j + 1] * 0.7154
	  + src->row_pointers[i][3*j + 2] * 0.0721;
      }
    } 
  }
}

/**
 * This is just a helper method. It calls specific functions that either run 
 * on the device (runKernel and run_const_shared_mem_kernel()) or the host (runSequential).
 * If the input image is colored, the function converts it to grayscale and then
 * passes it as input to one of the functions that does edge detection.
 */
void prepareKernelCall() {
  // input image
  frame_ptr from = input_frames[0];
  
  // Allocate frame for kernel to store its results into
  output_frames[0] = allocate_frame(from->image_height,
				    from->image_width,
				    1);
  // do grayscale conversion if the image contains
  // values for RGB colored image.
  if (input_frames[0]->num_components > 1) {
    // allocate a new frame for a grayscale image with height
    // and width similar to the input image.
    output_frames[1] = allocate_frame(from->image_height,
				      from->image_width,
				      1);

    // convert to grayscale, write it to output_frames[1]
    toGrayScale(input_frames[0], output_frames[1]);
    
    destroy_frame(input_frames[0]); // destroy old frame
    input_frames[0] = output_frames[1]; // put the new gray frame as input frame_ptr
    output_frames[1] = NULL; // clear out the output frames.
    output_frames[0] = input_frames[0];
  }
    
  // call a simple kernel without constant or shared memory, the sequential implementation,
  // or a constant memory kernel, or a constant memory with shared memory kernel.

  // this calls the regular device kernel. To do step by step, #define RUN_INDIVIDUAL STEPS,
  // comment out the unnecessary kernels, and copy the output of the kernel desired to the
  // parameter passed in.
  // runKernel(output_frames[0]);

  // this either runs a shared memory with constant memory kernel or a kernel with only constant
  // memory. #define RUNS_CONST_SHARED_MEM_KERNEL if you want the kernel with both optimizations.
  run_const_shared_mem_kernel(output_frames[0]);

  // this simply runs the sequential version of the program.
  // runSequential(output_frames[0]);
}


/*****************************************************************************
 ********************** SEQUENTIAL CODE BEGINS HERE **************************
 *****************************************************************************/

/**
 * A sequential implementation of a guassian blurring algorithm. it uses a 
 * (2k+1)*(2k+1) gaussian mask to do a convolution on the image and calculate
 * blurred pixels.
 * @param from an input image
 * @param to where the output is write.
 * @param kernel is the gaussian mask.
 * @param k is the integer described in the size of the gaussian mask.
 **/
void seq_gaussianBlur(frame_ptr from, frame_ptr to, float *kernel, int k) {
  // iterates over the entire image matrix and apply the
  // gaussian mask over the entire image.
  for (int row = 0; row < from->image_height; row++) {
    for (int col = 0; col < from->image_width; col++) {
      // blurred pixel.
      int newpixel = 0;

      // applying convolution with the gaussian mask.
      for (int i = -1*k; i <= k; i++) {
	int k_offset = (i+k) * (2*k + 1);
	  
	for (int j = -1*k; j <= k; j++) {
	  int nrow = row + i;
	  int ncol = col + j;

	  // make sure you are convolving over valid pixels
	  if (nrow >= 0 && ncol >= 0 &&
	      nrow < from->image_height && ncol < from->image_width) {
	    newpixel = newpixel + kernel[k_offset + (j+k)]
	      * from->image_buffer[nrow*from->image_width + ncol];
	  }
	}
      }

      // write the blurred pixel to the output image.
      to->image_buffer[row*from->image_width + col] = newpixel;
    }
  }
}

/**
 * applies sobel operators on the input image to generate magnitude matrix
 * and gradient angle, which are used on the next step to do non-maximum
 * suppression. 
 * @requires from, magnitude, angle have the same dimensions.
 * @requires xGradient, and yGradient have a 3x3 size.
 * @param from is the input image
 * @param magnitude is the image pointer where pixels gradient magnitude is written to.
 * @param angle is where gradient direction is written to.
 * @param xGradient, yGradient are sobel kernels in the x and y directions respectively.
 **/
void seq_gradientCalculation(frame_ptr from, frame_ptr magnitude, frame_ptr angle,
			     int * xGradient, int *yGradient) {
  // accumulates gradient in the x and y direction for each pixel
  int xGrad, yGrad;

  // iterates over the entire pixels of the image.
  for (int row = 0; row < from->image_height; row++) {
    for(int col = 0; col < from->image_width; col++) {
      // resets the accumulated gradient for each pixel
      xGrad = 0;
      yGrad = 0;

      // convolution of gradient masks with the pixel (row, col) region
      for (int i = -1; i <= 1; i++) {
	for (int j = -1; j <= 1; j++) {
	  int nrow = row + i;
	  int ncol = col + j;

	  // make sure the neighbor exists before applying convolution.
	  if ((nrow >= 0) && (ncol >= 0) &&
	      (nrow < from->image_height) && (ncol < from->image_width)) {
	    xGrad = xGrad + (xGradient[(i+1)*3 + (j+1)] * from->image_buffer[nrow*from->image_width + ncol]);
	    yGrad = yGrad + (yGradient[(i+1)*3 + (j+1)] * from->image_buffer[nrow*from->image_width + ncol]);
	  }
	}
      }

      // normalize pixel intensity values that are out of bounds (> 255 or < 0)
      if (xGrad > 255)
	xGrad = 255;
      if (yGrad > 255)
	yGrad = 255;
      
      xGrad = abs(xGrad);
      yGrad = abs(yGrad);

      // calculate the magnitude gradient and adds it to the output magnitude
      // image.
      int mag = hypot((float) xGrad, (float) yGrad);
      magnitude->image_buffer[row*from->image_width + col] = mag;

      // calculates the angle of each pixel, converts them to degrees and
      // write the result to the angle frame_ptr
      float angle_radians = atan2((float) yGrad, (float) xGrad);
      int angle_degrees = abs(angle_radians) * (180.0 / M_PI);
      angle->image_buffer[row*from->image_width + col] = angle_degrees;
    }
  }
}

/**
 * implements non-maximum suppression on the magnitude pixels given the 
 * angle information in the argument. The output of this stage is written to
 * the output frame_ptr.
 * @requires same dimension for all input frame_ptrs.
 **/
void seq_maxSuppression(frame_ptr magnitude, frame_ptr angle_fptr, frame_ptr output) {
  int height = magnitude->image_height;
  int width = magnitude->image_width;

  // iterate over all the pixels in the image and for each pixel (row, col)
  // do a hysteresis analysis.
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int back_pixel, front_pixel;
      int pixel = row*width + col;
      int angle = angle_fptr->image_buffer[pixel];

      // chooses the direction of the angle and checks if
      // the pixel at (row, col) is a local maximum or not.
      // it is suppressed if it is not a local maximum, otherwise it is kept.
      if (angle > 0 && angle < 23) { // 0 degree angle
	back_pixel = (col-1 >= 0) ? magnitude->image_buffer[pixel-1] : 0;
	front_pixel = (col+1 < width) ? magnitude->image_buffer[pixel+1] : 0;

      } else if (angle >= 23 && angle < 68) { // 45 degree angle
	back_pixel = ((row+1) < height && (col-1) >= 0) ? magnitude->image_buffer[(row+1)*width + (col-1)] : 0;
	front_pixel = ((row-1) >= 0 && (col+1) < width) ? magnitude->image_buffer[(row-1)*width + (col+1)] : 0;

      } else if (angle >= 68 && angle < 113) { // 90 degree angle
	back_pixel = (row - 1 >= 0) ? magnitude->image_buffer[(row-1)*width + col] : 0;
	front_pixel = (row + 1 < height) ? magnitude->image_buffer[(row+1)*width + col] : 0;

      } else if (angle >= 113 && angle < 158) { // 135 degree angle
	back_pixel = (row-1 >= 0 && col-1 >= 0) ? magnitude->image_buffer[(row-1)*width + (col-1)] : 0;
	front_pixel = ((row+1) < height && (col+1) < width) ? magnitude->image_buffer[(row+1)*width + (col+1)] : 0;

      } else { // everything else is around 180 degrees.
	back_pixel = (col-1 >= 0) ? magnitude->image_buffer[pixel-1] : 0;
	front_pixel = (col+1 < width) ? magnitude->image_buffer[pixel+1] : 0;

      }

      // suppressing the pixel if it is not the global maximum
      // in the line described by its angle.
      if (magnitude->image_buffer[pixel] < back_pixel ||
	  magnitude->image_buffer[pixel] < front_pixel) {
	output->image_buffer[pixel] = 0;
      } else {
	output->image_buffer[pixel] = magnitude->image_buffer[pixel];
      }
    }
  }
}

/**
 * Combines double threshold analysis with edge tracking to finalize the
 * edge detection algorithm. 
 * @requires: input, final_output have same dimension.
 * @param: low_threshold, high_threshold are the two thresholds to consider for threshold
 *         analysis.
 **/
void seq_doubleThresholdAndHysteresis(frame_ptr input, frame_ptr final_output,
				  int low_threshold, int high_threshold) {
  int width = input->image_width;
  int height = input->image_height;

  // double threshold analysis to classify pixels into either
  // a strong edge or weak edge.
  // iterates over the entire pixels of the input frame_ptr
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int pixel = row*width + col;

      // if greater than the threshold, set it as a strong edge
      // else if between low and high threshold, set it as a weak edge
      // else suppress it.
      if (input->image_buffer[pixel] >= high_threshold) {
	input->image_buffer[pixel] = 255;
      } else if (input->image_buffer[pixel] < high_threshold &&
		 input->image_buffer[pixel] >= low_threshold) {
	input->image_buffer[pixel] = low_threshold;
      } else {
	input->image_buffer[pixel] = 0;
      }
    }
  }

  // hyteresis analysis to find the relationship between weak and
  // strong edges.
  // iterates over the entire pixels in the image.
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int pixel = row*width + col;
      
      // hysteresis edge tracking: we look at the neighbors of the weak pixel (row, col) and
      // if there is a strong neighbor, the pixel becomes strong.
      if (input->image_buffer[pixel] > 0 && input->image_buffer[pixel] < 255)
	{
	  // check to see if any of the 8 neighbors of the pixel (row, col)
	  // with weak intensity is a strong edge.
	  // make sure also there is a neighbor in the col-1, col+1, row-1, row+1 directions.
	  if (((col-1 >= 0) && (input->image_buffer[row*width + (col-1)] == 255)) ||
	      ((col+1 < width) && (input->image_buffer[row*width + (col+1)] == 255)) ||
	      ((row+1 < height) && (input->image_buffer[(row+1)*width + col] == 255)) ||
	      ((row+1 < height) && (col-1 >= 0) && input->image_buffer[(row+1)*width + (col-1)] == 255) ||
	      ((row+1 < height) && (col+1 < width) && (input->image_buffer[(row+1)*width + (col+1)] == 255)) ||
	      ((row-1 >= 0) && (col+1 < width) && (input->image_buffer[(row-1)*width + (col+1)] == 255)) ||
	      ((row-1 >= 0) && (col-1 >= 0) && (input->image_buffer[(row-1)*width + (col-1)] == 255)) ||
	      ((row-1 >= 0) && (input->image_buffer[(row-1)*width + col] == 255)))
	    {
	      final_output->image_buffer[pixel] = 255;
	    }
	  else
	    {
	      final_output->image_buffer[pixel] = 0;
	    }
	}
      else {
	final_output->image_buffer[pixel] = input->image_buffer[pixel];
      }
    }
  }
}


/**
 * Runs the sequential implementation of the Canny Edge algorithm.
 * it creates the necessary temporary frame_ptr for each step of the
 * algorithm and reuse some frame_ptr as is fit.
 **/
void runSequential(frame_ptr final_output) {
  printf("\t..... Running Sequential......\n");
  // calculates the elapse time for the function.
  clock_t time_in_milli;
  time_in_milli = clock();
  
  frame_ptr greyimage = input_frames[0];
  frame_ptr blurimage = allocate_frame(greyimage->image_height,
				       greyimage->image_width, 1);
  frame_ptr magnitude = allocate_frame(greyimage->image_height,
				       greyimage->image_width, 1);
  // kernel mask for gaussian filter
  int k = 2;
  float kernel[GAUSSIAN_MASK_SIZE];
  float sigma = SIGMA;
  set_convo_kernel(kernel, k, sigma);

  // blurs the image to remove noise.
  seq_gaussianBlur(greyimage, blurimage, kernel, k);

  // calculate gradient changes in the image to find edges
  // reuses greyimage frame_ptr to store pixels angles
  seq_gradientCalculation(blurimage, magnitude, greyimage, xGradient, yGradient);
  
  // non-maximum suppression
  // reusing blurimage frame_ptr as output for the maximum suppression
  // operation.
  seq_maxSuppression(magnitude, greyimage, blurimage);

  // hysteresis analysis-- edge tracking to find the relationship between
  // weak and strong edges.
  // blurimage refers to maxSuppressed output.
  seq_doubleThresholdAndHysteresis(blurimage, final_output, low_threshold, high_threshold);

  time_in_milli = clock() - time_in_milli;
  double inMilli = (((double) time_in_milli) / CLOCKS_PER_SEC) * 1000; 
  printf("Elapsed Time in Milliseconds: %f\n", inMilli);
  
  // kill the frames allocated here
  destroy_frame(blurimage);
  destroy_frame(magnitude);
}

/******************************************************************************
 ********************** DEVICE CODE STARTS HERE *******************************
 ******************************************************************************/

/**
 * A cuda implementation of a guassian blurring algorithm. it uses a
 * (2k+1)*(2k+1) gaussian mask to do a convolution on the image and calculate
 * blurred pixels. No constant memory or shared memory is used here.
 * @param from an input image
 * @param to where the output is write.
 * @param kernel is the gaussian mask.
 * @param k is the integer described in the size of the gaussian mask.
 **/
__global__ void APPLY_GAUSSIAN_BLUR(float *kernel, int k, unsigned char *from,
				    unsigned char  *to, int height, int width)
{
  int row, col, newpixel, k_len;
  
  newpixel = 0; // the new blurred pixel.
  k_len = 2*k + 1; // length of the kernel mask.
  col = threadIdx.x + blockIdx.x * blockDim.x;
  row = threadIdx.y + blockIdx.y * blockDim.y;

  // make sure it is a valid pixel.
  if (col < width && row < height) {
    for (int i = -1*k; i <= k; i++) { // iterates kernel row
      int k_offset = (i+k) * k_len;
      
      for (int j = -1*k; j <= k; j++) { // iterates kernel col
	int nrow = row + i;
	int ncol = col + j;

	// make sure the neighbor being considered for convolution actually exists.
	if (nrow >= 0 && ncol >= 0 && nrow < height && ncol < width) {
	  newpixel = newpixel + kernel[k_offset + (j+k)] * from[nrow*width + ncol];
	}
      }
    }
    // writes the blurred pixel to the "to" frame_ptr.
    to[row*width + col] = newpixel;
  }
}

/**
 * Applies a Sobel filter mask (either in the x or y direction) in a convolution over the neighbors of each pixel
 * handled by each thread. 
 * @requires: from and to have the same dimension specified by height and width.
 * @requires: sobelKernel is 3x3
 **/
__global__ void applySobelOperatorKernel(int *sobelKernel, unsigned char *from, unsigned char *to,
					 int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int newPixel = 0;
  
  // make sure it is a valid pixel.
  if (col < width && row < height) {
    // convolve the sobel operator kernel with
    // the pixels next to the pixel (row,col)
    // and saves the new result to (row,col)
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
	int nrow = row + i;
	int ncol = col + j;
	// bounds checking.
	if (nrow >= 0 && ncol >= 0 && nrow < height && ncol < width) {
	  newPixel = newPixel + sobelKernel[(i+1)*3 + (j+1)] * from[nrow*width + ncol];
	}	
      }
    }

    // normalize the out of bounds pixel values (> 255 or < 0)
    if (newPixel < 0) {
      newPixel = abs( newPixel);
    }
    if (newPixel > 255) {
      newPixel = 255;
    }

    // write it to the output
    to[row*width + col] = newPixel;
  }  
}

/**
 * Given the gradient matrix in the x and y direction, this function computes the gradient magnitude
 * and angle which are used for non-maximum suppression. 
 * @requires: Gx, Gy, magnitude, and pixel_angle have the same dimensions, specified by the height, width.
 **/
__global__ void pixelMagnitudeAndAngle(unsigned char *Gx, unsigned char *Gy, unsigned char *magnitude,
			     unsigned char *pixel_angle, int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // make sure it is a valid pixel
  if (col < width && row < height) {
    int pixel = row*width + col;
    magnitude[pixel] = hypot((float) Gx[pixel], (float) Gy[pixel]);
    
    // gets the angle of the pixel and collapses it to the nearest horizontal, vertical,
    // or diagonal angle of (0, 45, 90, 135 degrees angles) by converting the angle
    // in radian to degrees. It also gets the absolute value of the angle to reduce redundancy
    // of two points on the opposite ends of the same diagonal line. That way we have, pi/2 and -pi/2
    // mapping to pi/2, pi and 0 mapping to 0, etc.
    float arctan = atan2((float) Gy[pixel], (float) Gx[pixel]);
    float inDegrees = abs(arctan) * (180.0 / M_PI);

    // collapses the different angles into four categories depending on the
    // proximity of the angle found to each of the four.
    if (inDegrees > 0 && inDegrees <= 22.5) {
      pixel_angle[pixel] = 0;
    } else if (inDegrees > 22.5 && inDegrees <= 67.5) {
      pixel_angle[pixel] = 45;
    } else if (inDegrees > 67.5 && inDegrees <= 112.5) {
      pixel_angle[pixel] = 90;
    } else if (inDegrees > 112.5 &&  inDegrees <= 157.5) {
      pixel_angle[pixel] = 135;
    } else { // because we get absolute value, everything else is either 180 or 0
      pixel_angle[pixel] = 0;
    }
  }
}

/**
 * Given the pixel gradient angle information, this function suppresses pixels that 
 * are not local maximum in the direction dictated by their angle. This is the 
 * non-maximum analysis stage. 
 * @requires: all image matrix inputs have the dimension described in height and width.
 **/
__global__ void nonMaximumSuppression(unsigned char *magnitude, unsigned char *pixel_angle,
				      unsigned char *final_suppression, int height, int width)
{
  int front_pixel, back_pixel, pixel, row, col;
  
  col = threadIdx.x + blockIdx.x*blockDim.x;
  row = threadIdx.y + blockIdx.y*blockDim.y;

  // make sure it is a valid pixel.
  if (col < width && row < height) {
    pixel = row*width + col;

    // chooses a back and front neighbor based on whether the neighbor
    // in the direction given by pixel angle exists.
    if (pixel_angle[pixel] == 0) {
      back_pixel = (col-1 >= 0) ? magnitude[pixel-1] : 0;
      front_pixel = (col+1 < width) ? magnitude[pixel+1] : 0;
    } else if (pixel_angle[pixel] == 45) {
      back_pixel = ((row+1) < height && (col-1) >= 0) ? magnitude[(row+1)*width + (col-1)] : 0;
      front_pixel = ((row-1) >= 0 && (col+1) < width) ? magnitude[(row-1)*width + (col+1)] : 0;
    } else if (pixel_angle[pixel] == 90) {
      back_pixel = (row - 1 >= 0) ? magnitude[(row-1)*width + col] : 0;
      front_pixel = (row + 1 < height) ? magnitude[(row+1)*width + col] : 0;
    } else if (pixel_angle[pixel] == 135) {
      back_pixel = (row-1 >= 0 && col-1 >= 0) ? magnitude[(row-1)*width + (col-1)] : 0;
      front_pixel = ((row+1) < height && (col+1) < width) ? magnitude[(row+1)*width + (col+1)] : 0;
    } else {
      printf("### BAD ANGLE: %d\n", pixel_angle[pixel]);
    }

    // suppressing the pixel if it is not the global maximum
    // in the line described by its angle.
    if (magnitude[pixel] < back_pixel ||
	magnitude[pixel] < front_pixel) {
      final_suppression[pixel] = 0;
    } else {
      final_suppression[pixel] = magnitude[pixel];
    }
  }
}

/**
 * Given low and high thresholds, this function suppresses or keep pixels based on whether
 * they are greater the low_threshold or not. It standardizes all strong edges here.
 * @requires: image pixels have the dimension described by height and width.
 **/
__global__ void thresholdAnalysis(unsigned char *suppressed_pixels, unsigned char *output_pixels,
				  int high_threshold, int low_threshold, int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // make sure its a valid pixel.
  if (col < width && row < height) {
    int pixel = row*width + col;

    // suppress less than the low threshold. Standardize to strong edge if
    // greater than high threshold.
    if (suppressed_pixels[pixel] >= high_threshold)
      output_pixels[pixel] = 255;
    else if (suppressed_pixels[pixel] < high_threshold &&
	     suppressed_pixels[pixel] >= low_threshold)
      output_pixels[pixel] = low_threshold;
    else
      output_pixels[pixel] = 0;
  }
}

/**
 * Does hysteresis analysis to find relationship between weak edges and strong edges.
 * the output of this step is the final output of the edge detection algorithm.
 * @requires: dimension of image matrices are equal to height * width.
 **/
__global__ void hystEdgeTracking(unsigned char *threshold_pixels, unsigned char *output_pixels,
				 int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // checks if its a valid pixel.
  if (col < width  && row < height) {
    // hysteresis edge tracking: we look at the neighbors of the pixel (row, col) and
    // if there is a strong neighbor, the pixel becomes strong.
    int pixel = row*width + col;

    // check if it is a weak edge or not.
    if (threshold_pixels[pixel] > 0 && threshold_pixels[pixel] < 255) {
      int found = 0;
      // check the neighbors of the weak edge to find if there is
      // a strong edge around.
      for (int i = -1; i <= 1; i++) {
	for (int j = -1; j <= 1; j++) {
	  int nrow = row + i;
	  int ncol = col + j;

	  // make sure the neighbor exists.
	  if (nrow >= 0 && ncol >= 0 && nrow < height && ncol < width)
	    if (threshold_pixels[nrow*width + ncol] == 255) {
	      found = 1;
	      i = j = 3;
	    }
	  // declare the weak edge strong if it has a strong neighbor.
	  if (found)
	    output_pixels[pixel] = 255;
	  else
	    output_pixels[pixel] = 0;
	}
      }
    } else {
      output_pixels[pixel] = threshold_pixels[pixel];
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////

/**
 * A regular implementation for finding the gradient and angle using sobel operators.
 * Here instead of doing it one by one as above, we do everything together so that we
 * the optimized implementation seen in the sequential, constant memory, and shared memory
 * implementations.
 * @requires: the same specification as the other implementation above.
 ***/
__global__ void gradient_calculation(unsigned char *blurMatrix, unsigned char *magMatrix,
				     unsigned char *angleMatrix, int *xGradient, int *yGradient,
				     int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // make sure its a valid pixel
  if (col < width && row < height) {
    int xGrad = 0;
    int yGrad = 0;

    // convolve the sobel operator kernel with
    // the pixels next to the pixel (row,col)
    // and saves the new result to (row,col)
    for (int i = -1; i <= 1; i++) {
      // saves the row index here
      int nrow = row + i;

      for (int j = -1; j <= 1; j++) {
	int ncol = col + j;
	if (ncol >= 0 && nrow >= 0 && ncol < width && nrow < height) {
	  xGrad = xGrad + xGradient[(i+1)*3 + (j+1)] * blurMatrix[nrow*width + ncol];
	  yGrad = yGrad + yGradient[(i+1)*3 + (j+1)] * blurMatrix[nrow*width + ncol];
	}
      }
    }
    // saves the magnitude gradient value.
    magMatrix[row*width + col] = hypot((float) xGrad, (float) yGrad);

    // finds the pixel angle in degrees.
    float angle_radians = atan2((float) yGrad, (float) xGrad);
    int angle_degrees = abs(angle_radians) * (180.0 / M_PI);
    angleMatrix[row*width + col] = angle_degrees;
  }
}

/**
 * A generic device non-maximum suppression algorithm. It is used for all three different
 * implementations. No need for constant memory or shared memory. It is an alternative to
 * the regular device implementation(nonMaxSuppression) which depends on collapsing angles 
 * to either 0, 45, 90, or 135 degrees.
 * @requires: the parameters conform to the specification in nonMaxSuppression.
 **/
__global__ void non_max_suppression(unsigned char *magMatrix, unsigned char *angleMatrix,
				    unsigned char *suppressedMatrix, int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // make sure its a valid pixel
  if (col < width && row < height) {
    int back_pixel, front_pixel;
    int pixel = row*width + col;
    int angle = angleMatrix[pixel];

    // chooses the direction of the angle and checks if
    // the pixel at (row, col) is a local maximum or not.
    // it is suppressed if it is not a local maximum, otherwise it is kept.
    if (angle > 0 && angle < 23) { // 0 degree angle
      back_pixel = (col-1 >= 0) ? magMatrix[pixel-1] : 0;
      front_pixel = (col+1 < width) ? magMatrix[pixel+1] : 0;
      
    } else if (angle >= 23 && angle < 68) { // 45 degree angle
      back_pixel = ((row+1) < height && (col-1) >= 0) ? magMatrix[(row+1)*width + (col-1)] : 0;
      front_pixel = ((row-1) >= 0 && (col+1) < width) ? magMatrix[(row-1)*width + (col+1)] : 0;
      
    } else if (angle >= 68 && angle < 113) { // 90 degree angle
      back_pixel = (row - 1 >= 0) ? magMatrix[(row-1)*width + col] : 0;
      front_pixel = (row + 1 < height) ? magMatrix[(row+1)*width + col] : 0;
      
    } else if (angle >= 113 && angle < 158) { // 135 degree angle
      back_pixel = (row-1 >= 0 && col-1 >= 0) ? magMatrix[(row-1)*width + (col-1)] : 0;
      front_pixel = ((row+1) < height && (col+1) < width) ? magMatrix[(row+1)*width + (col+1)] : 0;
      
    } else { // everything else is around 180 degrees.
      back_pixel = (col-1 >= 0) ? magMatrix[pixel-1] : 0;
      front_pixel = (col+1 < width) ? magMatrix[pixel+1] : 0;
      
    }

    // suppressing the pixel if it is not the global maximum
    // in the line described by its angle.
    if (magMatrix[pixel] < back_pixel ||
	magMatrix[pixel] < front_pixel) {
      suppressedMatrix[pixel] = 0;
    } else {
      suppressedMatrix[pixel] = magMatrix[pixel];
    }
  }
}

/*****************************************************************************
 ********************* WITH CONSTANT MEMORY AND CACHING **********************
 *****************************************************************************/

/**
 * Constant memory for sobel operators mask and gaussian kernel mask.
 **/
__constant__ int xGradientMask[SOBEL_MASK_SIZE];
__constant__ int yGradientMask[SOBEL_MASK_SIZE];
__constant__ float GaussianMask[GAUSSIAN_MASK_SIZE];

/**
 * A guassian blur implementation that uses constant memory (GaussianMask) declared above.
 * It requires that the kernel mask generated for the convolution is copied to the constant
 * memory.
 * Everything else happens like the regular device gaussian implementation above.
 **/
__global__ void const_mem_gaussian_blur(unsigned char *inputMatrix, unsigned char *blurMatrix,
					int k, int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int blurPixel = 0;
  int kernelLen = 2*k + 1;

  // make sure it is a valid pixel.
  if (col < width && row < height) {
    // do convolution by iterating over all the neighbors
    // of the pixel (row, col).
    for (int i = -1*k; i <= k; i++) {
      int nrow = row + i;
      int offset = (i+k) * kernelLen;
    
      for (int j = -1*k; j <= k; j++) {
	int ncol = col + j;

	// make sure the neighbor exists.
	if (ncol < width && ncol >= 0 && nrow < height && nrow >= 0) {
	  blurPixel = blurPixel + GaussianMask[offset + (j+k)] * inputMatrix[nrow*width + ncol]; 
	}
      }
    }
    // write the pixel output.
    blurMatrix[row*width + col] = blurPixel;
  }
}

/**
 * This function finds the gradient magnitude and gradient angle of the blurMatrix
 * by convolving sobelMasks (in the y and x directions) with the blurMatrix.
 * The gradient in the x and y directions are found in place to avoid using extra space
 * and computational time.
 **/
__global__ void const_mem_sobel_filter(unsigned char *blurMatrix, unsigned char *magMatrix,
				       unsigned char *angleMatrix, int height, int width)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // make sure its a valid pixel
  if (col < width && row < height) {
    int xGrad = 0;
    int yGrad = 0;

    // convolve the sobel operator kernel with
    // the pixels next to the pixel (row,col)
    // and saves the new result to (row,col)
    for (int i = -1; i <= 1; i++) {
      // saves the row index here
      int nrow = row + i;
     
      for (int j = -1; j <= 1; j++) {
	int ncol = col + j;
	if (ncol >= 0 && nrow >= 0 && ncol < width && nrow < height) {
	  xGrad = xGrad + xGradientMask[(i+1)*3 + (j+1)] * blurMatrix[nrow*width + ncol];
	  yGrad = yGrad + yGradientMask[(i+1)*3 + (j+1)] * blurMatrix[nrow*width + ncol];
	}
      }
    }
    // saves the magnitude gradient value.
    magMatrix[row*width + col] = hypot((float) xGrad, (float) yGrad);

    // finds the pixel angle in degrees.
    float angle_radians = atan2((float) yGrad, (float) xGrad);
    int angle_degrees = abs(angle_radians) * (180.0 / M_PI);
    angleMatrix[row*width + col] = angle_degrees;
  }
}

/****************************************************************************
 ********************** TILING AND SHARED MEMORY ****************************
 ****************************************************************************/

/**
 * Defines the relationship between tileWidth and BlockWidth for
 * the use of shared memory and pulling data from global memory.
 * blockWidth = tileWidth + kernelLength - 1;
 **/
#define SHARED_MEM_TILE_WIDTH 4
#define SHARED_MEM_FOR_SOBEL (SHARED_MEM_TILE_WIDTH + 2)
#define GAUSSIAN_LEN 5
#define SOBEL_LEN 3
#define G_LEN (SHARED_MEM_TILE_WIDTH + GAUSSIAN_LEN - 1);
#define S_LEN (SHARED_MEM_FOR_SOBEL + SOBEL_LEN - 1);

// for regular blocks-- without using tiling.
#define REG_BLOCK_LEN 32

// makes it compatible for CUDA. For some reason, couldnt
// assign the immediate #defines
__constant__ const int gausLen = G_LEN;
__constant__ const int sobelLen = S_LEN;

/**
 * This is an extension over the const_mem_gaussian_blur algorithm described above.
 * The only addition is the used of shared memory and tiling to locally saved 
 * global data into a shared memory.
 * @requires: something is copied to the GaussianMask constant memory.
 * @requires: the relationship between tileWidth and BlockWidth is preserved here.
 * @requires: dimension of matrices should match the height and width argument.
 **/
__global__ void const_shared_mem_gaussian_blur(unsigned char *inputMatrix, unsigned char *blurMatrix,
					       int k, int tileWidth, int height, int width)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col_o = blockIdx.x * tileWidth + tx; // index to the output
  int row_o = blockIdx.y * tileWidth + ty; // index to the output.

  int kernelLen = 2*k + 1;
  int col_i = col_o - (kernelLen/2); // where to draw the data for shared memory from.
  int row_i = row_o - (kernelLen/2); // where to draw the data for shared memory from.
  
  int blurPixel = 0;

  // shared memory per block.
  __shared__ unsigned char shared_tile[gausLen][gausLen];
  
  // retrieving data from the global memory to the shared tile memory.
  // makes sure a thread is working on a valid.
  if ((row_i >= 0) && (col_i >= 0) &&
      (row_i < height) && (col_i < width)) {
    shared_tile[ty][tx] = inputMatrix[row_i*width + col_i];
  } else {
    // put 0 in the place of invalid pixels.
    shared_tile[ty][tx] = 0;
  }

  __syncthreads();

  // make sure the thread is supposed to be doing computations.
  if (ty < tileWidth && tx < tileWidth) {
    // convolution happens here by iterating over the neighbors of
    // the pixel.
    for (int i = 0; i < kernelLen; i++)
      for (int j = 0; j < kernelLen; j++)
	blurPixel = blurPixel + GaussianMask[i*kernelLen + j]*shared_tile[i+ty][j+tx];

    // make sure the output indices are valid.
    if (row_o < height && col_o < width) {
      blurMatrix[row_o*width + col_o] = blurPixel;
    }
  }
}

/**
 * This is also an extension of the constant memory sobel filter device kernel above.
 * In addition to using constant memory, it also uses shared memory to compute the gradient
 * magnitude and direction from the blurMatrix. 
 * @requires: GaussianMask, and xGradientMask and yGradientMask have the appropriate data copied in.
 * @requires: the relationship between tileWidth and blockWidth is reserved.
 * @requires: dimensions match the arguments given.
 **/
__global__ void const_shared_mem_sobel_filter(unsigned char *blurMatrix, unsigned char *magMatrix,
					      unsigned char *angleMatrix, int k, int tileWidth,
					      int height, int width)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // output indices
  int col_o = blockIdx.x * tileWidth + tx;
  int row_o = blockIdx.y * tileWidth + ty;

  // input indices.
  int col_i = col_o - 1;
  int row_i = row_o - 1;

  // shared memory for the block
  __shared__ unsigned char shared_tile[sobelLen][sobelLen];

  // retrieving data from the global memory to the shared tile memory.
  // makes sure a thread is working on a valid.
  if ((row_i >= 0) && (col_i >= 0) &&
      (row_i < height) && (col_i < width)) {
    shared_tile[ty][tx] = blurMatrix[row_i*width + col_i];
  } else {
    // put 0 in the place of invalid pixels.
    shared_tile[ty][tx] = 0;
  }

  __syncthreads();

  // make sure indices are within tileWith
  if (ty < tileWidth && tx < tileWidth) {
    int xGrad = 0;
    int yGrad = 0;

    // convolve the sobel operator kernel with
    // the pixels next to the pixel (row,col)
    // and saves the new result to (row,col)
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
	xGrad = xGrad + xGradientMask[i*3 + j] * shared_tile[i+ty][j+tx];
	yGrad = yGrad + yGradientMask[i*3 + j] * shared_tile[i+ty][j+tx];
      }
    }

    if (row_o < height && col_o < width) {
      // saves the magnitude gradient value.
      magMatrix[row_o*width + col_o] = hypot((float) xGrad, (float) yGrad);

      // finds the pixel angle in degrees.
      float angle_radians = atan2((float) yGrad, (float) xGrad);
      int angle_degrees = abs(angle_radians) * (180.0 / M_PI);
      angleMatrix[row_o*width + col_o] = angle_degrees;
    }
  }
}

/******************************************************************************
 ********************** DEVICE CODE ENDS HERE *********************************
 ******************************************************************************/

/**
 * Computes a Gaussian convolutional kernel for a given size. A gaussian
 * kernel is given by:
 * => H_{i,j} = (1/2*pi*std_dev^2)*exp(-[(i-(k+1))^2 +  (j-(k+1))^2] / (2*std_dev^2))
 *
 * The result of the operation is written to the integer buffer given.
 * @param kernel is a 1-dimensional array to contain kernel values. Indexing into
 *        the array is given by "row*kernel_length + col".
 * @param kernel_length is the height/width of the kernel. For every kernel (height = width).
 * @param std_dev is the standard deviation of values to consider when averaging
 *        neighboring pixels.
 *
 * The idea for normalizing:
 * https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
 */
void set_convo_kernel(float *kernel, int k, float sigma)
{
  if (k < 1) {
    printf("For Gaussian kernel, k is undefined: %d\n", k);
    exit(1);
  }
  if (sigma <= 0) {
    printf("Standard Deviation < 0: %f\n", sigma);
    exit(1);
  }

  //initializes constants of the Gaussian kernel.
  int kernLen = 2*k + 1;
  float sigmaSqCons = 2 * sigma * sigma;
  float sigmaPiCons = 1.0 / (sigmaSqCons * M_PI);
  int k_inc = k + 1;
  
  // iterates and fills rows and columns of the kernel
  float sum = 0.0;
  for (int i = 1; i <= kernLen; i++) {
    int row_offset = (i - 1) * kernLen;
    for (int j = 1; j <= kernLen; j++) {
      int index = row_offset + (j - 1);
      float i_pow = pow(i - k_inc, 2.0);
      float j_pow = pow(j - k_inc, 2.0);
      
      float val = sigmaPiCons * exp(-1 * (i_pow + j_pow) / sigmaSqCons);
      sum = sum + val;
      kernel[index] = val;
    }
  }

  // Normalize the kernel
  for (int x = 0; x < kernLen; ++x)
    for (int y = 0; y < kernLen; ++y)
      kernel[x*kernLen + y] /= sum;
}

/**
 * Finds the strongest pixel intensity value in an image and sets the upper threshold as 
 * a 0.7 * highest_pixel_intensity.
 * @requires: pixels_ptr->num_components = 1.
 * @param pixels_ptr is an array of pixels for an image.
 **/
int maxPixelIntensity(frame_ptr imgPixels)
{
  int max = 0;
  for (int i = 0; i < imgPixels->image_height; i++) {
    for (int j = 0; j < imgPixels->image_width; j++) {
      if (imgPixels->row_pointers[i][j] > max)
	max = imgPixels->row_pointers[i][j];
    }
  }
  return max;
}

/**
 * allocates space on the device memory and writes the address to the 
 * memory location to the d_pointer.
 * @param: d_pointer is the location of the pointer to the memory allocated 
 *      on the device
 * @param: numBytes is the number of bytes to allocate on the device.
 **/
void setDevMemory(void **d_pointer, int numBytes) {
  if (cudaMalloc(d_pointer, numBytes) != cudaSuccess) {
    printf("### CANT SET DEVICE MEMORY: %p\n", d_pointer);
    exit(1);
  }
}

/**
 * Copies bytes from one memory location to another. The memory locations 
 * can either be on the device, host, or both. This is just a wrapper function
 * for the cuda implementation, cudaMemcpy().
 * @param: dst where the bytes should be copied to.
 * @param: src the location that contains the bytes.
 * @param: numBytes is how many bytes there is to copy.
 * @param: dir indicates whether to copy from device to device, device to host,
 *         host to device, or host to host. 
 **/
void cpyMemory(void *dst, void *src, int numBytes, cudaMemcpyKind dir) {
  if (cudaMemcpy(dst, src, numBytes, dir) != cudaSuccess) {
    printf("### MEM CPY FAIL : %p -> %p\n", src, dst);
    exit(1);
  }
}

/**
 * Checks if the last device kernel was successfully executed or not.
 **/
void checkErrorForLastKernel() {
  if (cudaGetLastError() != cudaSuccess) {
    printf("### Kernel Execution failed ###\n");
    exit(1);
  }
}

/**
 * This function allocates the necessary resources for executing a kernel
 * that uses constant memory and shared memory (using tiling) to do convolution
 * over image as it processes it for edge detection. 
 * -- xGradient, yGradient are declared as globalconstant memory
 * -- gaussianMask is also declared as global constant memory
 * Any thread working on convolution reads mask data from the global constant memory,
 * participate in tiling and pulling data from global memory, and write to the
 * global memory.
 * @param result is the output frame_ptr to write into.
 ***/
void run_const_shared_mem_kernel(frame_ptr result)
{
  // setting a gaussian convolution kernel as a 5x5.
  int k = 2;
  int kernel_len = 2*k + 1;
  float kernel[GAUSSIAN_MASK_SIZE];
  float sigma = SIGMA;
  set_convo_kernel(kernel, k, sigma);


  /**
   * timing kernel execution
   */   
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);

  // copies the local gaussian mask into the constant memory declared
  // globally.
  if (cudaMemcpyToSymbol(GaussianMask, kernel, GAUSSIAN_MASK_SIZE * sizeof(int))
      != cudaSuccess) {
    printf("Couldn't write to the global constant memory\n");
    exit(1);
  }
  if (cudaMemcpyToSymbol(xGradientMask, xGradient, 9 * sizeof(int))
      != cudaSuccess) {
    printf("Couldn't write to the global constant memory\n");
    exit(1);
  }
  if (cudaMemcpyToSymbol(yGradientMask, yGradient, 9 * sizeof(int))
      != cudaSuccess) {
    printf("Couldn't write to the global constant memory\n");
    exit(1);
  }
  

  // image matrix information and the device pointers to store
  // input matrix and output matrix.
  unsigned char *d_from, *d_to, *d_magMatrix;
  int height = input_frames[0]->image_height;
  int width = input_frames[0]->image_width;
  int size = height * width;

  // allocate the space for input and output on the device
  // and copy the host input to the device input.
  setDevMemory((void**) &d_from, size);
  setDevMemory((void**) &d_to, size);
  setDevMemory((void**) &d_magMatrix, size);
  cpyMemory(d_from, input_frames[0]->image_buffer, size, cudaMemcpyHostToDevice);

  // setting up tileWidth information for block and grid dimensions
  // of the kernel to execute.
  int tileWidth = SHARED_MEM_TILE_WIDTH;
  int tileWidthSobel = SHARED_MEM_FOR_SOBEL;
  
  // when doing gaussian filter, a 5x5
  int blockWidth = tileWidth + kernel_len - 1;
  dim3 dimBlock(blockWidth, blockWidth, 1);

  // when doing sobel filters, a 3x3 kernel
  int blockWidth_Sobel = tileWidthSobel + 2;
  dim3 sobelBlock(blockWidth_Sobel, blockWidth_Sobel, 1);

  // a grid for shared memory, and a regular grid for anything else.
  dim3 dimGrid((width - 1)/tileWidth + 1, (height - 1)/tileWidth + 1, 1);

#ifdef RUN_CONST_SHARED_MEM_KERNEL
  // the way to do regular gridding when using shared memory.
  dim3 regGrid((width - 1)/blockWidth + 1, (height - 1)/blockWidth + 1, 1);
  
  printf("\t......Running Const_Shared_Mem_Kernel.....\n");
  
  // launching a kernel to perform a guassian blur on the image input.
  const_shared_mem_gaussian_blur<<<dimGrid, dimBlock>>>
    (d_from, d_to, k, tileWidth, height, width);
  checkErrorForLastKernel();

  // launching a kernel that performs sobel gradient analysis and
  // writes the result of gradient magnitude and pixel angle into
  // the matrices d_magMatrix and d_from, respectively
  const_shared_mem_sobel_filter<<<dimGrid, sobelBlock /*dimBlock*/>>>
    (d_to, d_magMatrix, d_from, k, tileWidth, height, width);
  checkErrorForLastKernel();
#else
  blockWidth = REG_BLOCK_LEN;
  dim3 regGrid((width - 1)/blockWidth + 1, (height - 1)/blockWidth + 1, 1);
  dimBlock.x = blockWidth;
  dimBlock.y = blockWidth;
  
  printf("\t.....Running Const_Mem_Kernel.....\n");
  // launching a kernel to perform a gaussian blur with constant
  // memory but without shared memory.
  const_mem_gaussian_blur<<<regGrid, dimBlock>>>
    (d_from, d_to, k, height, width);
  checkErrorForLastKernel();

  // launching a kernel that performs sobel gradient analysis with constant
  // memory but without using shared memory.
  const_mem_sobel_filter<<<regGrid, dimBlock>>>
    (d_to, d_magMatrix, d_from, height, width);
#endif
  
  // calls the non maximum suppression algorithm for a regular non constant
  // non-shared memory implementation
  non_max_suppression<<<regGrid, dimBlock>>>
    (d_magMatrix, d_from, d_to, height, width);
  checkErrorForLastKernel();
  
  thresholdAnalysis<<<regGrid, dimBlock>>>
    (d_to, d_from, high_threshold, low_threshold, height, width);
  checkErrorForLastKernel();

  // final step. calls the regular hysteresis analysis.
  hystEdgeTracking<<<regGrid, dimBlock>>>
    (d_from, d_to, height, width);
  checkErrorForLastKernel();

  cpyMemory(result->image_buffer, d_to, size, cudaMemcpyDeviceToHost);

  // synchronizing the start and stop times to get the
  // elapsed time.

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_inMilli = 0;
  cudaEventElapsedTime(&time_inMilli, start, stop);
  
  // prints the elapsed time.
  printf("Kernel Elapsed Time: %.8f\n", time_inMilli);

  cudaFree(d_from);
  cudaFree(d_magMatrix);
  cudaFree(d_to);
}

// This sets up GPU device by allocating the required memory and then
// calls the kernel on GPU. (You might choose to add/remove arguments.)
// It's currently set up to use the global variables and write its
// final results into the specified argument.
void runKernel(frame_ptr result)
{
  // testing set_convo_kernel
  int k = 2;
  int kernel_len = 2*k + 1;
  float kernel[GAUSSIAN_MASK_SIZE];
  float sigma = SIGMA;
  set_convo_kernel(kernel, k, sigma);
  float total = 0.0;
  
  for (int i = 0; i < kernel_len; i++) {
    for (int j = 0; j < kernel_len; j++) {
      total = total + kernel[i*kernel_len + j];
      printf("%5d", (int) round(159 * kernel[i*kernel_len + j]));
    }
    printf("\n");
  }
  printf("Gaussian Total: %.5f\n", total);


  /////////////////////////////////////////////////////////////
  unsigned char *d_from, *d_to, *d_final_to, *d_magnitude, *d_pixel_angle, *d_final_suppression;
  int height = input_frames[0]->image_height;
  int width = input_frames[0]->image_width;
  int size = height * width;

  printf("\t......Running Regular Kernel......\n");
  
  // cudaEvents to record the elapse time for kernel execution.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // allocates to and from frame_ptrs and copies the global
  // frame_ptrs into the device memory. Exits when cudaMalloc/cudaMemcpy fails.
  setDevMemory((void**) &d_from, size);
  setDevMemory((void**) &d_to, size);
  setDevMemory((void**) &d_final_to, size);
  setDevMemory((void**) &d_magnitude, size);
  setDevMemory((void**) &d_pixel_angle, size);
  setDevMemory((void**) &d_final_suppression, size);
  
  cpyMemory(d_from, input_frames[0]->image_buffer, size, cudaMemcpyHostToDevice);

  /// allocates space for the kernel weights and copies the
  /// kernel computed to the device memory.
  float *d_kernel;
  int k_numBytes = (kernel_len * kernel_len * sizeof(d_kernel[0]));

  setDevMemory((void**) &d_kernel, k_numBytes);
  cpyMemory(d_kernel, kernel, k_numBytes, cudaMemcpyHostToDevice);

  ////////////////////////////////////////////////////////////////
  // sets the block and grid dimensions.
  int block_side = REG_BLOCK_LEN;
  dim3 dimBlock(block_side, block_side, 1);
  dim3 dimGrid(ceil(width/ (float) block_side), ceil(height/ (float) block_side), 1);
  
  // kernel call to blur an image.
  APPLY_GAUSSIAN_BLUR<<<dimGrid, dimBlock>>>(d_kernel, k, d_from, d_to, height, width);
  checkErrorForLastKernel();

 
#ifdef RUNNING_INDIVIDUAL_STEPS
  // copies the result of Gaussian filter into the
  // from pointer to start the gradient kernel
  int g_numBytes = 9 * sizeof(xGradient[0]);
  int *d_sobelKernel;
  
  /// allocates space for a sobel kernel and copies one of the gradient kernels.
  setDevMemory((void**) &d_sobelKernel, g_numBytes);
  cpyMemory(d_sobelKernel, xGradient, g_numBytes, cudaMemcpyHostToDevice);

  /// copies the result of the first kernel as input to the second kernel.
  cpyMemory(d_from, d_to, size, cudaMemcpyDeviceToDevice);

  // Kernel call to apply Sobel Operator  
  applySobelOperatorKernel
    <<<dimGrid, dimBlock>>>(d_sobelKernel,
			    d_from,
			    d_to,
			    height, width);
  checkErrorForLastKernel();
  
  cpyMemory(d_sobelKernel, yGradient, 9 * sizeof(yGradient[0]), cudaMemcpyHostToDevice);

  // Kernel call to apply Sobel Operator  
  applySobelOperatorKernel
    <<<dimGrid, dimBlock>>>(d_sobelKernel,
			    d_from,
			    d_final_to,
			    height, width);
  checkErrorForLastKernel();

  // immediately frees gradient space allocated.
  cudaFree(d_sobelKernel);

  // gradient magnitude and angle analysis
  pixelMagnitudeAndAngle
    <<<dimGrid, dimBlock>>>(d_to,
			    d_final_to,
			    d_magnitude,
			    d_pixel_angle,
			    height, width);
  checkErrorForLastKernel();
  
  // non-maximum suppression analysis.
  nonMaximumSuppression
    <<<dimGrid, dimBlock>>>(d_magnitude,
			    d_pixel_angle,
			    d_final_suppression,
			    height, width);
  checkErrorForLastKernel();
#else
  // copies the result of Gaussian filter into the
  // from pointer to start the gradient kernel
  int g_numBytes = 9 * sizeof(xGradient[0]);
  int *d_xGradient, *d_yGradient;
  
  /// allocates space for the two sobel kernel and copies one of the gradient kernels.
  setDevMemory((void**) &d_xGradient, g_numBytes);
  setDevMemory((void**) &d_yGradient, g_numBytes);
  cpyMemory(d_xGradient, xGradient, g_numBytes, cudaMemcpyHostToDevice);
  cpyMemory(d_yGradient, yGradient, g_numBytes, cudaMemcpyHostToDevice);

  // calculates gradient and angle information in one phase.
  gradient_calculation<<<dimGrid, dimBlock>>>
    (d_to, d_magnitude, d_pixel_angle, d_xGradient, d_yGradient, height, width);
  checkErrorForLastKernel();

  // non maximum suppression using the angle and magnitude found above.
  non_max_suppression<<<dimGrid, dimBlock>>>
    (d_magnitude, d_pixel_angle, d_final_suppression, height, width);
  checkErrorForLastKernel();
  
  cudaFree(d_xGradient);
  cudaFree(d_yGradient);
#endif  
  /// double threshold analysis. 
  thresholdAnalysis
    <<<dimGrid, dimBlock>>>(d_final_suppression, d_magnitude,
			    high_threshold, low_threshold,
			    height, width);
  checkErrorForLastKernel();

  // hysteresis analysis - edge tracking to find relationship
  // between weak edges and strong edges.
  hystEdgeTracking
    <<<dimGrid, dimBlock>>>(d_magnitude,
			    d_to,
			    height, width);
  checkErrorForLastKernel();
  
  cudaEventRecord(stop);

  // copies the results from the device memory into the
  // host output frame_ptr
  cpyMemory(result->image_buffer, d_to, size, cudaMemcpyDeviceToHost);
  
  // synchronizing the start and stop times to get the
  // elapsed time.
  cudaEventSynchronize(stop);
  float time_inMilli = 0;
  cudaEventElapsedTime(&time_inMilli, start, stop);

  // prints the elapsed time.
  printf("Kernel Elapsed Time: %.8f\n", time_inMilli);

  // frees device resources
  cudaFree(d_from);
  cudaFree(d_to);
  cudaFree(d_kernel);
  cudaFree(d_final_to);
  cudaFree(d_pixel_angle);
  cudaFree(d_magnitude);
  cudaFree(d_final_suppression);
}

/**
 * Host main routine
 */
int main(int argc, char **argv) {
  if(argc < 3){
    fprintf(stderr, "ERROR: Need to specify input file and then output file\n");
    exit(1);
  }

  input_frames[0] = read_JPEG_file(argv[1]);   // Load input file
  prepareKernelCall();               // Do the actual work including calling CUDA kernel
  write_JPEG_file(argv[2], output_frames[0], 75);   // Write output file
  //runKernel(NULL);
  
  return 0;
}
