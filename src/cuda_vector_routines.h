/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#include <cuda_runtime.h>
#include <math.h>

// PBC routines

inline __device__ float4 min_image(float4 r,float lbox,float hbox)
{
	// assuming no more than one box away
	if (r.x > hbox) {
		r.x -= lbox;
	} else if (r.x < -hbox) {
		r.x += lbox;
	}
	if (r.y > hbox) {
		r.y -= lbox;
	} else if (r.y < -hbox) {
		r.y += lbox;
	}
	if (r.z > hbox) {
		r.z -= lbox;
	} else if (r.z < -hbox) {
		r.z += lbox;
	}
	return r;
}

inline __device__ float4 wrap(float4 r,float lbox)
{
	// assuming no more than one box away
	if (r.x > lbox) {
		r.x -= lbox;
	} else if (r.x < 0.0f) {
		r.x += lbox;
	}
	if (r.y > lbox) {
		r.y -= lbox;
	} else if (r.y < 0.0f) {
		r.y += lbox;
	}
	if (r.z > lbox) {
		r.z -= lbox;
	} else if (r.z < 0.0f) {
		r.z += lbox;
	}
	return r;
}

/////////////////////////////////////////
// 
/////////////////////////////////////////


inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}


////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    // NOTE: removed addition of fourth term on purpose
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w);
    // NOTE: removed addition to fourth term
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
//    a.w += b; // NOTE: removed addition of fourth term
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
//    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    // NOTE: removed fourth term
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    //a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    // NOTE: removed fourth term
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
        return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
        //return make_float4(__fdividef(a.x, b.x), __fdividef(a.y, b.y), __fdividef(a.z, b.z),  __fdividef(a.w, b.w));
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
        //a.x = __fdividef(a.x,b.x);
        //a.y = __fdividef(a.y,b.y);
        //a.z = __fdividef(a.z,b.z);
        a.x /= b.x;
        a.y /= b.y;
        a.z /= b.z;
        //a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
        return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
        //return make_float4(__fdividef(a.x, b), __fdividef(a.y, b), __fdividef(a.z, b),  __fdividef(a.w, b));
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
        //a.x = __fdividef(a.x,b);
        //a.y = __fdividef(a.y,b);
        //a.z = __fdividef(a.z,b);
        
        a.x /= b;
        a.y /= b;
        a.z /= b;
        //a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
        return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
        //return make_float4(__fdividef(b, a.x), __fdividef(b, a.y), __fdividef(b, a.z),  __fdividef(b, a.w));

}
