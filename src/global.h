#ifndef GLOBAL_H
#define GLOBAL_H

// Arythmetic Macros
#define SIGN(x)      (((x) < 0.0) ? -1.0 : 1.0)
#define SQR(x)       ((x) * (x))
#define CUBE(x)      ((x) * (x) * (x))

template<short N>
using vec_t = double[N];

#endif