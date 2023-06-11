// C++ program to check if three
// points are collinear or not
// using area of triangle.
#include <bits/stdc++.h>
#include <math.h>
#include <stdlib.h>

using namespace std;
// function to check if point
// collinear or not
void collinear(int x1, int y1, int x2,
               int y2, int x3, int y3)
{
	// Calculation the area of
	// triangle. We have skipped
	// multiplication with 0.5
	// to avoid floating point
	// computations
	int a = x1 * (y2 - y3) +
    x2 * (y3 - y1) +
    x3 * (y1 - y2);

	if (a == 0)
		cout << "Yes" << endl;
	else
		cout << "No" << endl;
}

void collinear_2(int x1, int y1, int x2,
                 int y2, int x3, int y3)
{
  if ((y3 - y2) * (x2 - x1) ==
      (y2 - y1) * (x3 - x2))
    cout << "Yes" << endl;
  else
    cout << "No" << endl;
}

// Driver Code
int main()
{
	// int x1 = 1, x2 = 1, x3 = 1,
	// 	y1 = 4, y2 = 1, y3 = 5;
	int x1 = 2, x2 = 1, x3 = 3,
		y1 = 2, y2 = 1, y3 = 3;
	collinear(x1, y1, x2, y2, x3, y3);
	collinear_2(x1, y1, x2, y2, x3, y3);
	return 0;
}
