# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

# include <GL/gl.h>
# include <GL/glu.h>
# include <GL/glut.h>

void life_init();
void life_update();

GLint window_width = 600;
GLint window_height = 600;
GLfloat left = 0.0;
GLfloat right = 1.0;
GLfloat bottom = 0.0;
GLfloat top = 1.0;
GLint game_width = 100;
GLint game_height = 100;

double pAlive = .20;

char gridA[100*100];
char gridB[100*100];
char *grid = gridA;
char *nextGrid = gridB;

#define WHITE 1.0, 1.0, 1.0
#define BLACK 0.0, 0.0, 0.0

int mod (int a, int b)
{
   if(b < 0) 
     return mod(-a, -b);   
   int ret = a % b;
   if(ret < 0)
     ret+=b;
   return ret;
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	
    GLfloat xSize = (right - left) / game_width;
	GLfloat ySize = (top - bottom) / game_height;
	
	GLint x,y;	

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

void update() {
	char *tempGrid;

    life_update();

	tempGrid = grid;
	grid = nextGrid;
	nextGrid = tempGrid;

	glutPostRedisplay();
    glutTimerFunc(1000 / 1, update, 0);
}

void graphics_init()
{
	
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Game of Life");
	glClearColor(1, 1, 1, 1);
	
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
}

int main(int argc, char **argv)
{
  glutInit(&argc, argv);
  graphics_init();

  // Initialize array
  life_init();

  update();
  glutMainLoop();

  return 0;
}

void life_init()
{
  int i, j;

  // Seed the random number generator
  srand(time(NULL));

  // Allocate memory for array
  //grid = ( char * ) malloc ( ( game_height ) * ( game_width ) * sizeof ( char ) );

  // initialize array, with each pixel having a probability of pAlive of being set to 1
  for( i = 0; i < game_height; i++ )
  {
    for( j = 0; j < game_width; j++ )
    {
	 // if( rand() <  pAlive * ((double)RAND_MAX + 1.0) )
     //{
     //   grid[i*game_width+j] = 1;
     // }
     // else
     // {
        gridA[i*game_width+j] = 0;
		gridB[i*game_width+j] = 0;
     // }
    }
  }

i = 2;
j = 1;
grid[i*game_width+j] = 1;

i = 1;
j = 2;
grid[i*game_width+j] = 1;

i = 0;
j = 0;
grid[i*game_width+j] = 1;

i = 0;
j = 1;
grid[i*game_width+j] = 1;

i = 0;
j = 2;
grid[i*game_width+j] = 1;

  //return grid;
}

void life_update()
{
  int i,j,aliveCount;
  for( i = 0; i < game_height; i++ )
  {
    for( j = 0; j < game_width; j++ )
    { 
      aliveCount = 0;
      aliveCount =
	    grid[ mod((i-1), game_height)*game_width + mod((j-1),  game_width) ] +
        grid[ mod((i),   game_height)*game_width + mod((j-1),  game_width) ] +
        grid[ mod((i+1), game_height)*game_width + mod((j-1),  game_width) ] + 
	    grid[ mod((i-1), game_height)*game_width + mod((j),    game_width) ] +
        grid[ mod((i+1), game_height)*game_width + mod((j),    game_width) ] +
	    grid[ mod((i-1), game_height)*game_width + mod((j+1),  game_width) ] +
        grid[ mod((i),   game_height)*game_width + mod((j+1),  game_width) ] +
        grid[ mod((i+1), game_height)*game_width + mod((j+1),  game_width) ]; 

		if(aliveCount > 0)
			printf("i: %d, j: %d, count: %d\n", i,j,aliveCount);

		if(grid[i * game_width + j] > 0)
			printf("grid[%d][%d]: %d\n", i,j,grid[i * game_width + j]);


      if( grid[i*game_width+j] == 1 )
      {
        if(aliveCount < 2 || aliveCount > 3)
          nextGrid[i*game_width+j] = 0;
        else
          nextGrid[i*game_width+j] = 1;
      }
      else
      {
        if(aliveCount == 3)
          nextGrid[i*game_width+j] = 1;
        else
          nextGrid[i*game_width+j] = 0;
      }
    }
  }
}
  
