#ifndef HILBERT3D_DEFINED

/*
	This is just a wrapper for the wonderfully speedy algorithm & code of J. K. Lawder!
	
	See the conjugate .cpp file for full copyright info.
	
	Todo:
		- consistent data types for parameters etc vs main code. We use simple ints here, which could cause problems for larger systems.
*/


class Hilbert3D
{
	public:
	
		Hilbert3D();
		~Hilbert3D();
		
		int CoordsToIndex( int x, int y, int z );
		void IndexToCoords( int index, int &x, int &y, int &z );
//		friend ostream& operator<<(ostream& os, Hilbert3D &hilbertCoord);
};


#define HILBERT3D_DEFINED
#endif
