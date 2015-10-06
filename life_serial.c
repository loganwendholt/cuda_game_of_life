# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

char * life_init(int, double, int, int);
void life_update(char *, int, int);

int main()
{
 
  char *grid;
  int rows = 480;
  int columns = 640;
  double pAlive = .20;
 
  // Initialize array
  grid = life_init(time(NULL), pAlive, rows, columns);

  while(1==1)
  {
    // Update array
    life_update(grid, rows, columns);
  }

  return 0;
}

char * life_init(int seed, double pAlive, int rows, int columns)
{
  char *grid;
  int i, j;

  // Seed the random number generator
  srand(seed);

  // Allocate memory for array
  grid = ( char * ) malloc ( ( rows ) * ( columns ) * sizeof ( char ) );

  // initialize array, with each pixel having a probability of pAlive of being set to 1
  for( i = 0; i < rows; i++ )
  {
    for( j = 0; j < columns; j++ )
    {
	  if( rand() <  pAlive * ((double)RAND_MAX + 1.0) )
      {
        grid[i*columns+j] = 1;
      }
      else
      {
        grid[i*columns+j] = 0;
      }
    }
  }

  return grid;
}

void life_update(char *grid, int rows, int columns)
{
  int i,j,aliveCount;
  for( i = 0; i < rows; i++ )
  {
    for( j = 0; j < columns; j++ )
    { 
	  aliveCount =
	    grid[ ((i-1) % rows)*columns + ((j-1) % columns) ] +
        grid[ ((i)   % rows)*columns + ((j-1) % columns) ] +
        grid[ ((i+1) % rows)*columns + ((j-1) % columns) ] + 
	    grid[ ((i-1) % rows)*columns + ((j)   % columns) ] +
        grid[ ((i+1) % rows)*columns + ((j)   % columns) ] +
	    grid[ ((i-1) % rows)*columns + ((j-1) % columns) ] +
        grid[ ((i)   % rows)*columns + ((j-1) % columns) ] +
        grid[ ((i+1) % rows)*columns + ((j-1) % columns) ]; 

      if( grid[i*columns+j] == 1 )
      {
        if(aliveCount < 2 || aliveCount > 3)
          grid[i*columns+j] = 0;
        else
          grid[i*columns+j] = 1;
      }
      else
      {
        if(aliveCount == 3)
          grid[i*columns+j] = 1;
        else
          grid[i*columns+j] = 0;
      }
    }
  }
}
  
