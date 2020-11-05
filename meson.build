project('Associated Legendre Polynomials', 'cpp')
gsl_dep = dependency('gsl')
executable('associated_legendre_polynomials', 'associated_legendre_polynomials.cpp', cpp_args : ['-std=c++17', '-DUSE_GSL'], dependencies: [gsl_dep])
