# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

/* Include the OpenGL headers */
# include <GL/gl.h>
# include <GL/glu.h>
# include <GL/glut.h>

/* ----- Function Declarations ----- */
void life_init();
void life_update();
int mod (int a, int b);
void display();
void reshape(int w, int h);
void update();
void graphics_init();
void add_rabbits_pattern(int start_x, int start_y);
struct timespec timer_start();
long timer_end(struct timespec start_time);

/* ----- Defines ----- */
#define WINDOW_WIDTH 640    // width of window in pixels
#define WINDOW_HEIGHT 480   // height of window in pixels
#define GAME_WIDTH 128      // number of cells it takes to span the window horizontally
#define GAME_HEIGHT 96      // number of cells it takes to span the window vertically

#define NUM_BLOCKS GAME_HEIGHT
#define NUM_THREADS GAME_WIDTH

#define WHITE 1.0, 1.0, 1.0 // RGB float values for white, in OpenGL format
#define BLACK 0.0, 0.0, 0.0 // RGB float values for black, in OpenGL format

/* ----- Global Variables ----- */

// Convert the #define parameters to OpenGL-compatible types
GLint window_width = WINDOW_WIDTH;
GLint window_height = WINDOW_HEIGHT;
GLint game_width = GAME_WIDTH;
GLint game_height = GAME_HEIGHT;

// Define the OpenGL vertices for the display area 
GLfloat left = 0.0;
GLfloat right = 1.0;
GLfloat bottom = 0.0;
GLfloat top = 1.0;

// Define the two arrays that contain all cell data
// These arrays will be used in a "ping-pong" fashion,
// where one array will be displayed while another array
// is being filled with updated data 
char gridA[GAME_HEIGHT*GAME_WIDTH];
char gridB[GAME_HEIGHT*GAME_WIDTH];

// define pointers to allow the grids to be easily swapped
char *grid = gridA;
char *nextGrid = gridB; 

// define pointers to GPU memory
char *lifeData;
char *nextLifeData;


/* ----- Function Definitions ----- */

/* ***************************************************
*  FUNCTION:  life_kernel
*
*  DESCRIPTION:
*    CUDA kernel to process a single cell on the life grid
*  
*  PARAMETERS:
*    char *sourceGrid:  a reference to the current simulation data
*    char *destGrid:    location of memory to save new simulation data
*
*  RETURN VALUE:
*    none
*
* ****************************************************/
__global__ void life_kernel(char *sourceGrid, char *destGrid)
{

  /* Work out our thread id */
  unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int tid = idx + idy * blockDim.x * gridDim.x;

  /* Get the x-y coordinates of the cell being processed */
  unsigned int x = tid % GAME_WIDTH;
  unsigned int y = tid / GAME_WIDTH;

  unsigned int xLeft  = (x-1 + GAME_WIDTH)  % GAME_WIDTH;
  unsigned int xRight = (x+1 + GAME_WIDTH)  % GAME_WIDTH;
  unsigned int yUp    = (y-1 + GAME_HEIGHT) % GAME_HEIGHT;
  unsigned int yDown  = (y+1 + GAME_HEIGHT) % GAME_HEIGHT;

  /* Count the number of live neighbors */
  aliveCount = 
	    sourceGrid[ yUp   * GAME_WIDTH + xLeft  ] +
        sourceGrid[ y     * GAME_WIDTH + xLeft  ] +
        sourceGrid[ yDown * GAME_WIDTH + xLeft  ] + 
	    sourceGrid[ yUp   * GAME_WIDTH + x      ] +
        sourceGrid[ yDown * GAME_WIDTH + x      ] +
	    sourceGrid[ yUp   * GAME_WIDTH + xRight ] +
        sourceGrid[ y     * GAME_WIDTH + xRight ] +
        sourceGrid[ yDown * GAME_WIDTH + xRight ]; 

  /* Calculate the next state of the cell */
  destGrid[tid] = aliveCount == 3 || (aliveCount == 2 && sourceGrid[tid]) ? 1 : 0;

}

/* ***************************************************
*  FUNCTION:  gpu_init
*
*  DESCRIPTION:
*    Allocate memory on the GPU 
*  
*  PARAMETERS:
*    none
*     
*  RETURN VALUE:
*    none
*
* ****************************************************/
void gpu_init()
{
  /* Allocate memory on the GPU */
  cudaMalloc(&lifeData, GAME_WIDTH*GAME_HEIGHT);
  cudaMalloc(&nextLifeData, GAME_WIDTH*GAME_HEIGHT);

  /* Transfer data to the GPU */
  cudaMemcpy( lifeData,      grid, GAME_WIDTH*GAME_HEIGHT, cudaMemcpyHostToDevice );
  cudaMemcpy( nextLifeData,  nextGrid, GAME_WIDTH*GAME_HEIGHT, cudaMemcpyHostToDevice );

/* ***************************************************
*  FUNCTION:  runLifeKernel
*
*  DESCRIPTION:
*    Run one step of the simulation through the CUDA kernel
*  
*  PARAMETERS:
*    none
*     
*  RETURN VALUE:
*    none
*
* ****************************************************/
void runLifeKernel()
{
  /* run kernel */
  life_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(lifeData, nextLifeData);

  /* transfer results from GPU */
  cudaMemcpy(grid, nextLifeData, GAME_WIDTH*GAME_HEIGHT, cudaMemcpyDeviceToHost );

  /* Swap the GPU arrays */
  char *tempData;
  tempData = lifeData;
  lifeData = nextLifeData;
  nextLifeData = temp;
}

/* ***************************************************
*  FUNCTION:  timer_start
*
*  DESCRIPTION:
*    Begins running a nanosecond-resolution timer
*  
*  PARAMETERS:
*    none
*     
*  RETURN VALUE:
*    struct timespec:  a reference to the timer's start time
*
* ****************************************************/
struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

/* ***************************************************
*  FUNCTION:  timer_end
*
*  DESCRIPTION:
*    Used to end a nanosecond-resolution timer
*  
*  PARAMETERS:
*    struct timespec start_time: the value returned by the timer_start function
*     
*  RETURN VALUE:
*    long:  The difference in the start and end times, in nanoseconds
*
* ****************************************************/
long timer_end(struct timespec start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    long diffInNanos = end_time.tv_nsec - start_time.tv_nsec;
    return diffInNanos;
}
// ----- END TIMING FUNCTIONS ------------------------



/* ***************************************************
*  FUNCTION:  display
*
*  DESCRIPTION:
*    Uses OpenGL to display the current array pointed
*    to by the global variable *grid
*  
*  PARAMETERS:
*    none
*     
*  RETURN VALUE:
*    none
*
* ****************************************************/
void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	
    GLfloat xSize = (right - left) / game_width;
	GLfloat ySize = (top - bottom) / game_height;
	
	GLint x,y;	
    // iterate through the grid and display each cell
    // as either a white or a black square (quad)
	glBegin(GL_QUADS);
	for (x = 0; x < game_width; ++x) {
		for (y = 0; y < game_height; ++y) {
            grid[x+y*game_width]?glColor3f(BLACK):glColor3f(WHITE);
            
			glVertex2f(    x*xSize+left,    y*ySize+bottom);
			glVertex2f((x+1)*xSize+left,    y*ySize+bottom);
			glVertex2f((x+1)*xSize+left,(y+1)*ySize+bottom);
			glVertex2f(    x*xSize+left,(y+1)*ySize+bottom);
		}
	}
	glEnd();
    	
	glFlush();
	glutSwapBuffers();
}


/* ***************************************************
*  FUNCTION:  reshape
*
*  DESCRIPTION:
*    Callback function used to handle any reshaping of the 
*    display window
*  
*  PARAMETERS:
*    int w:    new width
*    int h:    new height
*     
*  RETURN VALUE:
*    none
*
* ****************************************************/
void reshape(int w, int h) {
	window_width = w;
	window_height = h;

	glViewport(0, 0, window_width, window_height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(left, right, bottom, top);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutPostRedisplay();
}

/* ***************************************************
*  FUNCTION:  update
*
*  DESCRIPTION:
*    Callback function that calls the Game of Life 
*    siimulation function, then updates the screen
*    with the results.
*  
*  PARAMETERS:
*    none
*     
*  RETURN VALUE:
*    none
*
* ****************************************************/
void update() {
	struct timespec vartime = timer_start();
    runLifeKernel();
	long time_elapsed_nanos = timer_end(vartime);
	printf("Time taken (nanoseconds): %ld\n", time_elapsed_nanos);
	glutPostRedisplay();
}

/* ***************************************************
*  FUNCTION:  graphics_init
*
*  DESCRIPTION:
*    Handles window initialization and binding of
*    OpenGL/glut callback functions
*  
*  PARAMETERS:
*    none
*     
*  RETURN VALUE:
*    none
*
* ****************************************************/
void graphics_init()
{
	
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Game of Life");
	glClearColor(1, 1, 1, 1);
	
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
    glutIdleFunc(update);
}

/* ***************************************************
*  FUNCTION:  add_rabbits_pattern
*
*  DESCRIPTION:
*    Adds the "rabbits" pattern to the current grid
*  
*  PARAMETERS:
*    int start_x:  the x origin of the pattern (starting from left)
*    int start_y:  the y origin of the pattern (starting from top)
*
*  RETURN VALUE:
*    none
*
* ****************************************************/
void add_rabbits_pattern(int start_x, int start_y)
{
  int rabbits_pattern[18] = {0,0, 4,0, 5,0, 6,0, 0,1, 1,1, 2,1, 5,1, 1,2};
  int x, y, i;
  
  for(i=0; i<18; i+=2)
  {
    x = mod(rabbits_pattern[i] + start_x, game_width);
    y = mod(rabbits_pattern[i+1] + start_y, game_height);

    grid[y*game_width+x] = 1;
  }
}


/* ***************************************************
*  FUNCTION:  life_init
*
*  DESCRIPTION:
*    Initializes the arrays used to keep track of 
*    cell data. Zeros out both "cell buffers", then
*    applies a pattern to generate some life
*  
*  PARAMETERS:
*    none
*
*  RETURN VALUE:
*    none
*
* ****************************************************/
void life_init()
{
  int i, j;

  // Allocate memory for array
  //grid = ( char * ) malloc ( ( game_height ) * ( game_width ) * sizeof ( char ) );

  // zero out both buffers
  for( i = 0; i < game_height; i++ )
  {
    for( j = 0; j < game_width; j++ )
    {
      gridA[i*game_width+j] = 0;
      gridB[i*game_width+j] = 0;
    }
  }

  // add a pattern to the buffer
  add_rabbits_pattern(game_width/2-3,game_height/2-3);

}

/* ***************************************************
*  FUNCTION:  main
*
*  DESCRIPTION:
*    Main program execution loop. All actual looping
*    is handled by glut, so all calculations that
*    need to occur within the loop must be set within
*    glut callbacks. 
*  
*  PARAMETERS:
*    int argc:     Number of input parameters (unused)
*    char **argv:  Input parameters (unused)
*
*  RETURN VALUE:
*    int:  return status
*
* ****************************************************/
int main(int argc, char **argv)
{
  // Initialize OpenGL/GLUT
  glutInit(&argc, argv);
  graphics_init();

  // Initialize array
  life_init();

  // Initialize GPU
  gpu_init();

  // Begin main loop
  glutMainLoop();

  return 0;
}
  
