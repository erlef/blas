BLAS
=====
This project, funded by the [Erlang Ecosystem Foundation](https://erlef.org/), was made possible thanks to [Peerst Stritzinger](https://www.stritzinger.com/)
and aims to bring the efficiency of the BLAS-LAPACKE library to Erlang.

This implementation uses [openblas](https://github.com/xianyi/OpenBLAS). Other cblas/lapacke implementations can be used, but will need modifictations of c_src/Makefile linking options and c_src/eblas.h include options.

GRISP is supported, and uses the default [netlib](https://netlib.org) implementation.

A library reference is provided in [docu](docu/erlang_blas_ref.pdf).

INSTALLATION
=====
Make sure you have a BLAS-LAPACK library to link to. On ubuntu:
```
sudo apt-get install libopenblas-serial-dev
sudo apt-get install liblapacke-dev
```

This project can be added as a rebar3 dependency. Either clone it to a _checkout subfolder, or add it as a dependency in your rebar.config:
```
{deps,[blas]}.
```

SAFETY
=====
BLAS-LAPACKE executes in place over unbounded arrays, making sigsev crashes possible. This erlang library checks array dimensions for BLAS functions, but not for LAPACKE, making the latter potentially unsafe.

TESTING
=====
Basic tests cases are provided to the BLAS interface. However, LAPACKE is untested.

FUTURE WORK
====
Netlib's implementation of LAPACKE is provided with a test suite written in fortran. To use it, a [fortran interpreter]() was started, but could not be finished due to a lack of time. Another option tested was to write rewrite all LAPACKE function to pipe their arguments to an Erlang application which would execute them, and pipe the result back: [piper](https://github.com/tanguyl/piper). However, dealing with unbounded arrays proved harder than expected, piper could not be finished either. The latter option should be faster to finish, but a fortran interpreter for numerical computations could be a great tool to use existing fortran libraries.

Also, LAPACKE uses BLAS to perform its iterations. Using it in nifs present no advantages, and makes scheduling hard. Rewriting it in Erlang using a transpiler based upon the aforementioned fortran interpreter could solve this issue.
