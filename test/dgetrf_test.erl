-module(dgetrf_test).
-include_lib("eunit/include/eunit.hrl").

dgetrf_test()->
    A = blas:new(float64, [1, 2, -3, 1, 2, 4, 0, 7, -1, 3, 2, 0]),
    IPV = blas:new(int64, [0,0,0]),

    ok = blas:run({dgetrf, blasRowMajor, 4, 3, A, 4, IPV}),

    [2,3,3] = blas:btl(int64, blas:to_bin(IPV)),
    
    A_expected = [2, 4, 0, 1, -0.5, 5, 2, 7, 0.5, 0, -3, 0],
    A_expected =:= blas:btl(float64, blas:to_bin(A)).