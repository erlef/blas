blas
=====

This project is a NIF based wrapper of the Basic Linear Algebra Subprograms, which provides algorithms fine-tuned for a large variety of hardware.

This implementation uses [openblas](https://github.com/xianyi/OpenBLAS). Other cblas implementations can be used, but will need modifictations of c_src/Makefile linking options and c_src/eblas.h include options.

API
----
The following sites provide a clear BLAS API:
- [netlib's](https://netlib.org/blas/) concise reference.
- [intel's](https://www.intel.com/content/www/us/en/content-details/671183/developer-reference-for-intel-math-kernel-library-intel-mkl-11-3-c.html?wapkw=BLAS%20mlk) complete reference.
- [IBM's](https://www.ibm.com/docs/en) reference, with clear examples.

Usage
-----
```erlang
blas:run(Tuple);        % Execute on dirty scheduler
blas:run(Tuple, dirty); % Execute on dirty scheduler.
blas:run(Tuple, clean); % Execute on usual scheduler.
```
Tuple contains, in sequence, a BLAS function name (represented as an atom), followed by its arguments in Erlang representation.

Examples
-----

[caxpy](https://www.ibm.com/docs/en/essl/6.1?topic=vss-saxpy-daxpy-caxpy-zaxpy-multiply-vector-by-scalar-add-vector-store-in-vector): single Complex numbers, Alpha*x Plus Y:

```erlang
Alpha = blas:ltb(c, [1,0]),
X     = blas:ltb(c, [1,2,  2,3,  3,1]),
Y     = blas:new(c, [1,1,  0,0,  1,2]),
ok    = blas:run({caxpy, 3, Alpha, X, 1, Y, 1}),

io:format("Result: ~p~n", [blas:to_list(c, Y)]).
```

[stpmv](https://www.ibm.com/docs/en/essl/6.1?topic=mvs-strmv-dtrmv-ctrmv-ztrmv-stpmv-dtpmv-ctpmv-ztpmv-matrix-vector-product-triangular-matrix-its-transpose-its-conjugate-transpose): Single real numbers, Triangular Packed Matrix, Matrix*Vector operation.

```erlang
N  = 3,
A  = blas:new(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
X  = blas:new(s, [-0.25,-0.125,0.5]),
ok = blas:run(
    {stpmv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, A, X, 1},
    clean
),
[1.0,2.0,3.0] = blas:to_list(s, X).
```

Datatype conversion tables
-----

```
    enums       cblas[value]           blas[value]

    numbers
                (const)int             int
                void*                  c_binary
                const float            double; binary; c_binary
                const double           double; binary; c_binary
                const void*            binary; c_binary

```

For example, taking stpmv's cblas signature:

```c
void cblas_stpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST float *Ap, float *X, OPENBLAS_CONST blasint incX);
```
The arguments can be represented as such:
```erlang
% enums
Order  = blasRowMajor  || blasColMajor
Uplo   = blasUpper     || blasLower
Transa = blasNoTrans   || blasTrans
Diag   = blasNonUnit   || blasUnit

% int
N      = 3 
incX   = 1

% const float*
A      = blas:new(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); % c_binary, a mutable binary
A_2    = blas:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); % binary

% float*
X      = blas:new(s, [-0.25,-0.125,0.5]);             % c_binary
```

Creating c_binaries
-----
Since BLAS functions operate in place, c_binaries representing mutable arrays are needed. They can be created as such: 
```erlang
C_binary = blas:new(Binary).
% Binary is an erlang binary.
C_binary = blas:new(Type, List).
% List is a list of numbers.
% Type is used to encode the binary, and is one of: 
% s -> single precision
% d -> double precision
% c -> single complex numbers: List is of even size
% Z -> double complex numbers: List is of even size
```

Reading c_binaries
-----
The content of c_binaries can be retrieved as either a list of elements, or a constant binary.
Retrieving a list is done as such:
```erlang
C_binary = blas:to_list(Type, C_binary).
% Binary is a c_binary created using blas:new.
% Type is used to decode the binary, and is one of: 
% s -> single precision
% d -> double precision
% c -> single complex numbers (equivalent to s)
% Z -> double complex numbers (equivqlent to d)
```
Retrieving a binary is done as such:
```erlang
Binary = blas:to_bin(C_binary).
% Binary is a c_binary created using blas:new.
```
