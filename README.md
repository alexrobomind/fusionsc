**Poisson Disk Points Generator**

(C) Sergey Kosarevsky, 2014-2021

@corporateshark sk@linderdaum.com

http://www.linderdaum.com

http://blog.linderdaum.com

=============================

Poisson disk & Vogel disk points generator in a single file header-only C++11 library.

Usage example:
--------------
```
#define POISSON_PROGRESS_INDICATOR 1
#include "PoissonGenerator.h"
...
PoissonGenerator::DefaultPRNG PRNG;
const auto Points = PoissonGenerator::generatePoissonPoints( numPoints, PRNG );
...
const auto Points = PoissonGenerator::generateVogelPoints( numPoints );
```

Build instructions:
-----------

*gcc Poisson.cpp -std=c++11 -lstdc++*

*cmake -G "Visual Studio 16 2019" -A x64*

Demo app usage:
---------------
	Poisson [density-map-rgb24.bmp] [--raw-points] [--num-points=<value>] [--square] [--vogel-disk]

Algorithm description can be found in "Fast Poisson Disk Sampling in Arbitrary Dimensions"
http://people.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

Implementation is based on http://devmag.org.za/2009/05/03/poisson-disk-sampling/

=============================

Poisson disk

![Poisson disk](https://camo.githubusercontent.com/d2c51f5ef0b0644378910f2c58e3e29c3e297ab7/687474703a2f2f626c6f672e6c696e6465726461756d2e636f6d2f696d616765732f506f6973736f6e5f4469736b2e706e67)

Poisson rectangle

![Poisson rectange](https://camo.githubusercontent.com/bfd11990fb6d6050d372fc251c2a161344016175/687474703a2f2f626c6f672e6c696e6465726461756d2e636f6d2f696d616765732f506f6973736f6e5f526563742e706e67)

Poisson rectangle with custom density map

![Poisson rectangle with custom density map](https://camo.githubusercontent.com/82a7c564779471147872bfd35ee0c246b11eb8c3/687474703a2f2f626c6f672e6c696e6465726461756d2e636f6d2f696d616765732f506f6973736f6e5f526563745f44656e736974792e706e67)

Vogel disk

![Vogel disk](https://user-images.githubusercontent.com/2510143/144725686-59f31ade-8ddf-4461-9e18-eda0a7cd3146.png)
